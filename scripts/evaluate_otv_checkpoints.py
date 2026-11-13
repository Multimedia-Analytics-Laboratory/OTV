#!/usr/bin/env python3

"""Evaluate OTV checkpoints on a tokenized dataset."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


from llamafactory.model.adapter import _attach_otv_modules
from llamafactory.train.sft.trainer import (
    OTV_TRUTH_TOKEN,
    _get_first_attention_module,
    _select_verification_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved OTV checkpoints on a tokenized dataset.")
    parser.add_argument("--base_model", required=True, help="Base model path or Hugging Face model ID.")
    parser.add_argument("--run_dir", required=True, help="Training run directory containing OTV checkpoints.")
    parser.add_argument("--tokenized_path", required=True, help="Evaluation dataset saved by the preparation script.")
    parser.add_argument("--split", default="validation", help="Existing dataset split to evaluate.")
    parser.add_argument("--verifier_mode", choices=["hidden_state", "kv_cache"], required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--output_path", required=True, help="Output JSON file containing evaluation results.")
    parser.add_argument("--scores_dir", default="")
    return parser.parse_args()


def discover_checkpoints(run_dir: Path, include_final: bool) -> list[Path]:
    if (run_dir / "reg_model.safetensors").exists():
        return [run_dir]

    checkpoints = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            checkpoints.append(child)
    checkpoints.sort(key=lambda path: int(path.name.split("-")[-1]))
    if include_final:
        checkpoints.append(run_dir)
    return checkpoints


def load_regression_head_from_safetensors(head: torch.nn.Module, regression_path: Path) -> None:
    state_dict = {}
    with safe_open(str(regression_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            clean_key = key.replace("regression_head.", "", 1)
            state_dict[clean_key] = handle.get_tensor(key)
    head.load_state_dict(state_dict, strict=False)


def load_hidden_state_model(base_model: str, checkpoint_dir: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model = _attach_otv_modules(model, str(checkpoint_dir / "reg_model.safetensors"))
    model.regression_head = model.regression_head.to(model.device, dtype=torch.float32)
    model.eval()
    return model


def load_kv_cache_model(base_model: str, checkpoint_dir: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model = _attach_otv_modules(model, str(checkpoint_dir / "reg_model.safetensors"))
    model.regression_head = model.regression_head.to(model.device, dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    truth_token_id = tokenizer.convert_tokens_to_ids(OTV_TRUTH_TOKEN)
    if truth_token_id is None or truth_token_id < 0:
        raise ValueError(f"Missing truth token in tokenizer for {checkpoint_dir}")
    return model, tokenizer, truth_token_id


def roc_auc_score(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += index
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    total_pos = sum(labels)
    if total_pos == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ordered, start=1):
        if label == 1:
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / total_pos


def accuracy_at_threshold(labels: list[int], scores: list[float], threshold: float) -> float:
    correct = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        correct += int(pred == label)
    return correct / max(len(labels), 1)


def best_threshold_accuracy(labels: list[int], scores: list[float]) -> tuple[float, float]:
    candidates = {0.5}
    ordered = sorted(set(scores))
    if ordered:
        candidates.add(ordered[0] - 1e-6)
        candidates.add(ordered[-1] + 1e-6)
    for left, right in zip(ordered, ordered[1:]):
        candidates.add((left + right) / 2.0)
    best_acc = -1.0
    best_thr = 0.5
    for threshold in sorted(candidates):
        acc = accuracy_at_threshold(labels, scores, threshold)
        if acc > best_acc:
            best_acc = acc
            best_thr = threshold
    return best_acc, best_thr


def score_hidden_state(model, sample: dict) -> float:
    device = model.device
    input_ids = torch.tensor([sample["input_ids"]], device=device)
    attention_mask = torch.tensor([sample["attention_mask"]], device=device)
    trace_len = int(attention_mask.sum(dim=-1).item())
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids[:, :trace_len],
            attention_mask=attention_mask[:, :trace_len],
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1][:, trace_len - 1 : trace_len, :].float()
        score = model.regression_head(hidden)
    return float(score.item())


def score_kv_cache(model, tokenizer, truth_token_id: int, sample: dict) -> float:
    device = model.device
    input_ids = torch.tensor([sample["input_ids"]], device=device)
    attention_mask = torch.tensor([sample["attention_mask"]], device=device)
    trace_len = int(attention_mask.sum(dim=-1).item())
    trace_inputs = {
        "input_ids": input_ids[:, :trace_len],
        "attention_mask": attention_mask[:, :trace_len],
    }
    attention_cls = type(_get_first_attention_module(model))
    origin_forward = model.origin_forward
    verification_forward = _select_verification_forward(model)

    try:
        with torch.no_grad():
            with model.disable_adapter():
                attention_cls.forward = origin_forward
                generation_output = model.generate(
                    **trace_inputs,
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=1,
                )

            special_ids = list(getattr(tokenizer, "all_special_ids", []))
            if not special_ids and tokenizer.pad_token_id is not None:
                special_ids = [tokenizer.pad_token_id]

            if special_ids:
                special_ids_tensor = torch.tensor(special_ids, device=generation_output.sequences.device)
                special_mask = torch.isin(generation_output.sequences, special_ids_tensor)
                new_attention_mask = (~special_mask).to(generation_output.sequences.dtype)
            else:
                new_attention_mask = torch.ones_like(generation_output.sequences)

            new_ids = torch.full_like(generation_output.sequences, truth_token_id)
            attention_cls.forward = verification_forward
            outputs = model(
                input_ids=new_ids,
                attention_mask=new_attention_mask,
                past_key_values=generation_output.past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1][:, trace_len : trace_len + 1, :].float()
            score = model.regression_head(hidden)
    finally:
        attention_cls.forward = origin_forward

    return float(score.item())


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_dir: Path, dataset) -> tuple[dict, list[dict]]:
    device = "cuda:0"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.verifier_mode == "hidden_state":
        model = load_hidden_state_model(args.base_model, checkpoint_dir, device)
        tokenizer = None
        truth_token_id = None
    else:
        model, tokenizer, truth_token_id = load_kv_cache_model(args.base_model, checkpoint_dir, device)

    sample_rows = []
    labels = []
    scores = []
    max_samples = args.max_samples if args.max_samples is not None else len(dataset)

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        label = int(sample["otv_correct"])
        if args.verifier_mode == "hidden_state":
            score = score_hidden_state(model, sample)
        else:
            score = score_kv_cache(model, tokenizer, truth_token_id, sample)

        labels.append(label)
        scores.append(score)
        sample_rows.append(
            {
                "index": index,
                "label": label,
                "score": score,
                "trace_len": int(sum(sample["attention_mask"])),
                "response_start": int(sample.get("response_start", -1)),
            }
        )

    fixed_acc = accuracy_at_threshold(labels, scores, 0.5)
    best_acc, best_threshold = best_threshold_accuracy(labels, scores)
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]

    summary = {
        "checkpoint": checkpoint_dir.name,
        "path": str(checkpoint_dir),
        "verifier_mode": args.verifier_mode,
        "num_samples": len(labels),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "accuracy_at_0.5": fixed_acc,
        "best_threshold_accuracy": best_acc,
        "best_threshold": best_threshold,
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision(labels, scores),
        "mean_positive_score": (sum(positives) / len(positives)) if positives else None,
        "mean_negative_score": (sum(negatives) / len(negatives)) if negatives else None,
        "max_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "reserved_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, sample_rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tokenized_path = Path(args.tokenized_path)
    output_path = Path(args.output_path)
    scores_dir = Path(args.scores_dir) if args.scores_dir else None

    dataset_obj = load_from_disk(str(tokenized_path))
    if hasattr(dataset_obj, "keys"):
        if args.split in dataset_obj:
            dataset = dataset_obj[args.split]
        elif "train" in dataset_obj:
            dataset = dataset_obj["train"]
        else:
            first_key = next(iter(dataset_obj.keys()))
            dataset = dataset_obj[first_key]
    else:
        dataset = dataset_obj

    summaries = []
    for checkpoint_dir in discover_checkpoints(run_dir, include_final=args.include_final):
        summary, sample_rows = evaluate_checkpoint(args, checkpoint_dir, dataset)
        summaries.append(summary)

        if scores_dir is not None:
            scores_dir.mkdir(parents=True, exist_ok=True)
            scores_path = scores_dir / f"{checkpoint_dir.name}.jsonl"
            with scores_path.open("w", encoding="utf-8") as handle:
                for row in sample_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
