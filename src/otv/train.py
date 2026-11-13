"""Train the standalone verifier from completed, outcome-labeled JSONL traces."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import TraceCollator, TraceDataset
from .model import OTVVerifier
from .targets import TARGET_MODES, make_targets


def device_and_dtype(device="auto", dtype="auto"):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if torch.device(device).type not in ("cuda", "cpu"):
        raise ValueError("This reference implementation supports CPU or a single CUDA device.")
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
    if dtype == "bfloat16" and not device.startswith("cuda"):
        raise ValueError("Use float32 for the CPU reference run.")
    return device, {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]


def backward_batch(model, batch, chunk_size=128, target_mode="linear", loss_scale=1.0):
    """Accumulate the exact dense-target MSE gradient, freeing each probe chunk."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    cache = model.prefill(batch["input_ids"], batch["attention_mask"])
    count = (cache.lengths - batch["response_starts"] + 1).sum().item()
    total_loss = 0.0
    for left in range(int(batch["response_starts"].min()), cache.sequence_length + 1, chunk_size):
        positions = torch.arange(left, min(left + chunk_size, cache.sequence_length + 1), device=model.device)
        targets, valid = make_targets(cache.lengths, batch["response_starts"],
                                      batch["correct"], positions, target_mode)
        query_positions = positions[None].expand(cache.lengths.shape[0], -1)
        query_positions = torch.minimum(query_positions, cache.lengths[:, None])
        scores = model(cache, query_positions).float()
        loss = ((scores - targets).square() * valid).sum() / count
        (loss * loss_scale).backward()
        total_loss += loss.detach().item()
    return total_loss


def train_step(model, batch, optimizer, chunk_size=128, target_mode="linear", max_grad_norm=1.0):
    """One optimizer step, also used by the executable notebook."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = backward_batch(model, batch, chunk_size, target_mode)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(model, loader, chunk_size, target_mode):
    was_training = model.training
    squared_error, count = 0.0, 0
    model.eval()
    try:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            cache = model.prefill(batch["input_ids"], batch["attention_mask"])
            for left in range(int(batch["response_starts"].min()), cache.sequence_length + 1, chunk_size):
                positions = torch.arange(left, min(left + chunk_size, cache.sequence_length + 1), device=model.device)
                targets, valid = make_targets(cache.lengths, batch["response_starts"], batch["correct"], positions, target_mode)
                query = torch.minimum(positions[None].expand(cache.lengths.shape[0], -1), cache.lengths[:, None])
                scores = model(cache, query).float()
                squared_error += ((scores - targets).square() * valid).sum().item()
                count += valid.sum().item()
    finally:
        model.train(was_training)
    return squared_error / max(count, 1)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Base model path or Hugging Face ID.")
    p.add_argument("--train-data", required=True)
    p.add_argument("--eval-data")
    p.add_argument("--output", required=True, help="New or empty output directory.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--probe-chunk-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--head-dims", default="1024,256")
    p.add_argument("--head-dropout", type=float, default=.2)
    p.add_argument("--target-mode", choices=TARGET_MODES, default="linear")
    p.add_argument("--prompt-format", choices=("auto", "chat", "plain"), default="auto")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200,
                   help="Validation interval; zero evaluates only at each epoch/run end.")
    p.add_argument("--max-steps", type=int, default=0, help="Zero uses all epochs.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    for key in ("epochs", "batch_size", "gradient_accumulation", "probe_chunk_size", "max_length"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive.")
    if args.eval_steps < 0 or args.save_steps < 0 or args.max_steps < 0:
        raise ValueError("eval_steps, save_steps, and max_steps cannot be negative.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty --output directory.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype(args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    dataset = TraceDataset(args.train_data, tokenizer, args.max_length, args.prompt_format)
    collate = TraceCollator(pad_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    validation = None
    if args.eval_data:
        validation = DataLoader(TraceDataset(args.eval_data, tokenizer, args.max_length, args.prompt_format),
                                batch_size=args.batch_size, collate_fn=collate)
    model = OTVVerifier.from_pretrained(
        args.model, device=device, dtype=dtype, rank=args.rank, alpha=args.alpha,
        head_dims=tuple(int(size) for size in args.head_dims.split(",")),
        head_dropout=args.head_dropout,
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    updates = 0
    with (output / "metrics.jsonl").open("w") as metrics:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for index, batch in enumerate(loader):
                group_start = index // args.gradient_accumulation * args.gradient_accumulation
                divisor = min(args.gradient_accumulation, len(loader) - group_start)
                group_loss += backward_batch(model, batch, args.probe_chunk_size, args.target_mode, 1 / divisor) / divisor
                if (index + 1) % args.gradient_accumulation and index + 1 != len(loader):
                    continue
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                row = {"step": updates, "epoch": epoch + 1, "train_mse": group_loss}
                end_of_run = args.max_steps > 0 and updates >= args.max_steps
                end_of_epoch = index + 1 == len(loader)
                eval_due = args.eval_steps > 0 and updates % args.eval_steps == 0
                if validation is not None and (eval_due or end_of_epoch or end_of_run):
                    row["eval_mse"] = evaluate(model, validation, args.probe_chunk_size, args.target_mode)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
                group_loss = 0.0
                if args.save_steps > 0 and updates % args.save_steps == 0:
                    model.save_pretrained(output / f"checkpoint-{updates}", tokenizer)
                if args.max_steps > 0 and updates >= args.max_steps:
                    break
            if args.max_steps > 0 and updates >= args.max_steps:
                break
    model.save_pretrained(output, tokenizer)
    print(f"Saved standalone verifier to {output}")


if __name__ == "__main__":
    main()
