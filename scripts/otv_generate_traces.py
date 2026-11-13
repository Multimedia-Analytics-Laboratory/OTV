"""Generate reasoning traces and correctness labels for OTV training."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {path}.")


def maybe_apply_chat_template(tokenizer, questions: list[str], enable_thinking: bool) -> list[str]:
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def format_question(question: str, prefix: str, suffix: str) -> str:
    question = question.strip()
    parts = []
    if prefix:
        parts.append(prefix.strip())
    parts.append(question)
    if suffix:
        parts.append(suffix.strip())
    return "\n\n".join(parts)


def _clean_answer_text(text: str) -> str:
    text = text.strip()
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("：", ":")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_answer_prefixes(text: str) -> str:
    patterns = [
        r"(?i)^(?:final answer|answer|the answer is|thus the answer is|therefore the answer is|so the answer is|hence the answer is)\s*[:：]?\s*",
        r"^(?:答案|答案是|答え|答えは|最终答案|最終答案|解答|結論|结论)\s*[:：]?\s*",
    ]
    text = _clean_answer_text(text)
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", text).strip()
            if updated != text:
                text = updated
                changed = True
    return text


def _maybe_extract_simple_answer(text: str) -> str:
    text = _strip_answer_prefixes(text)

    variable_match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$", text)
    if variable_match:
        text = variable_match.group(1).strip()

    option_match = re.match(r"^(?:option\s+)?([A-D])(?:\b|$)", text, flags=re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()

    numeric_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if numeric_match:
        remaining = (text[: numeric_match.start()] + text[numeric_match.end() :]).strip()
        remaining = remaining.strip("()[]{}<>.,;:!?% 。")
        if remaining == "" or re.fullmatch(r"[A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", remaining):
            return numeric_match.group(0)

    return text.strip("()[]<>.,;:!? ")


def normalize_answer(text: str) -> str:
    text = text.strip()
    box_match = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if box_match:
        text = box_match[-1]

    lines = [_clean_answer_text(line) for line in text.splitlines() if line.strip()]
    answer_line_match = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        inline_match = re.match(
            r"(?i)^(?:final answer|answer|the answer is|thus the answer is|therefore the answer is|so the answer is|hence the answer is)\s*[:：]?\s*(.+?)\s*$",
            line,
        )
        if inline_match:
            candidate = inline_match.group(1).strip()
            if candidate.strip(":：*- ") != "":
                answer_line_match = candidate
                break

        localized_match = re.match(r"^(?:答案|答案是|答え|答えは|最终答案|最終答案|解答|結論|结论)\s*[:：]?\s*(.+?)\s*$", line)
        if localized_match:
            candidate = localized_match.group(1).strip()
            if candidate.strip(":：*- ") != "":
                answer_line_match = candidate
                break

    if answer_line_match is None:
        for index, line in enumerate(lines[:-1]):
            label_only_match = re.match(
                r"(?i)^(?:final answer|answer|答案|答案是|答え|答えは|最终答案|最終答案|解答|結論|结论)\s*[:：]?\s*$",
                line,
            )
            if label_only_match:
                answer_line_match = lines[index + 1]
                break

    if answer_line_match is not None:
        text = answer_line_match
    else:
        marker_match = re.findall(r"(?:####)\s*(.+)", text, flags=re.IGNORECASE)
        if marker_match:
            text = marker_match[-1]
        elif lines:
            text = lines[-1]

    text = _maybe_extract_simple_answer(text)
    text = text.replace("$", "").replace(",", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.。]+$", "", text)
    return text


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate labeled OTV traces from questions and reference answers.")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path or Hugging Face model ID.")
    parser.add_argument("--input_path", required=True, help="Input questions and reference answers in JSON or JSONL.")
    parser.add_argument("--output_path", required=True, help="Output JSONL file containing labeled traces.")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--question_field", default="query", help="Field containing the input question.")
    parser.add_argument("--answer_field", default="response", help="Field containing the reference answer.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question_prefix", default="")
    parser.add_argument("--question_suffix", default="")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def init_transformers_backend(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def init_vllm_backend(args):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "vLLM backend requested, but `vllm` is not installed in the current Python environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tokenizer": args.model_name_or_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max(args.max_prompt_length + args.max_new_tokens, args.max_prompt_length),
        "trust_remote_code": True,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    return tokenizer, LLM(**llm_kwargs)


def generate_transformers(args, model, tokenizer, prompts: list[str]) -> list[list[str]]:
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    if torch.cuda.is_available():
        model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            num_return_sequences=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    generated_only = generated[:, prompt_length:]
    generated_texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    grouped_outputs = []
    for prompt_index in range(len(prompts)):
        start = prompt_index * args.samples_per_prompt
        end = start + args.samples_per_prompt
        grouped_outputs.append(generated_texts[start:end])
    return grouped_outputs


def generate_vllm(args, llm, tokenizer, prompts: list[str]) -> list[list[str]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=args.samples_per_prompt,
        temperature=max(args.temperature, 1e-5) if args.temperature > 0 else 0.0,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    grouped_outputs = []
    for output in outputs:
        candidates = sorted(output.outputs, key=lambda item: item.index)
        grouped_outputs.append([candidate.text for candidate in candidates])
    return grouped_outputs


def main() -> None:
    args = build_argparser().parse_args()
    if args.samples_per_prompt < 1:
        raise ValueError("--samples_per_prompt must be at least 1.")
    if args.samples_per_prompt > 1 and args.temperature <= 0:
        raise ValueError("--samples_per_prompt > 1 requires sampling, so --temperature must be > 0.")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    records = records[args.start :: args.stride]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.backend == "transformers":
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        tokenizer, backend = init_transformers_backend(args)
    else:
        tokenizer, backend = init_vllm_backend(args)

    total_outputs = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start : batch_start + args.batch_size]
            questions = [
                format_question(item[args.question_field], args.question_prefix, args.question_suffix)
                for item in batch
            ]
            prompts = maybe_apply_chat_template(tokenizer, questions, enable_thinking=args.enable_thinking)
            if args.backend == "transformers":
                generated_groups = generate_transformers(args, backend, tokenizer, prompts)
            else:
                generated_groups = generate_vllm(args, backend, tokenizer, prompts)

            for batch_index, sample in enumerate(batch):
                formatted_question = questions[batch_index]
                raw_question = sample[args.question_field]
                raw_ground_truth = sample.get(args.answer_field, "")
                ground_truth = normalize_answer(sample.get(args.answer_field, ""))
                for sample_index in range(args.samples_per_prompt):
                    generated_text = generated_groups[batch_index][sample_index]
                    prediction = normalize_answer(generated_text)
                    record = {
                        "question": formatted_question,
                        "raw_question": raw_question,
                        "response": generated_text.strip(),
                        "raw_ground_truth": raw_ground_truth,
                        "ground_truth": ground_truth,
                        "predict": prediction,
                        "correct": int(prediction == ground_truth and prediction != ""),
                        "source_index": args.start + (batch_start + batch_index) * args.stride,
                        "sample_index": sample_index,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_outputs += 1

            handle.flush()

    print(f"Saved {total_outputs} traces to {output_path}")


if __name__ == "__main__":
    main()
