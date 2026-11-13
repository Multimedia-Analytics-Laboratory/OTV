"""Score response prefixes using a trained standalone OTV checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import encode_trace, load_records
from .model import OTVVerifier
from .train import device_and_dtype


@torch.no_grad()
def score_trace(model, tokenizer, question, response, chunk_size=128,
                tail_tokens=100, prompt_format="auto"):
    if chunk_size < 1 or tail_tokens < 1:
        raise ValueError("chunk_size and tail_tokens must be positive.")
    row = encode_trace(tokenizer, question, response, prompt_format=prompt_format)
    ids = torch.tensor([row["input_ids"]], device=model.device)
    was_training = model.training
    model.eval()
    try:
        cache = model.prefill(ids)
        positions = torch.arange(row["response_start"], ids.shape[1] + 1, device=model.device)
        chunks = [model(cache, pos).squeeze(0).float().cpu() for pos in positions.split(chunk_size)]
        scores = torch.cat(chunks)
    finally:
        model.train(was_training)
    # Exclude the empty-response boundary from trace-level aggregation.
    tail = scores[1:][-tail_tokens:]
    return {"trace_score": tail.mean().item(), "final_score": scores[-1].item(),
            "response_tokens_seen": list(range(len(scores))), "prefix_scores": scores.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", help="Override the base model recorded in the checkpoint.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL or JSON traces; correctness labels are optional.")
    source.add_argument("--question")
    p.add_argument("--response")
    p.add_argument("--output", help="Optional JSONL destination; defaults to stdout.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    p.add_argument("--probe-chunk-size", type=int, default=128)
    p.add_argument("--tail-tokens", type=int, default=100)
    p.add_argument("--prompt-format", choices=("auto", "chat", "plain"), default="auto")
    args = p.parse_args()
    if args.question is not None and args.response is None:
        p.error("--response is required with --question")
    records = load_records(args.input) if args.input else [{"question": args.question, "response": args.response}]
    device, dtype = device_and_dtype(args.device, args.dtype)
    model = OTVVerifier.from_pretrained(args.model, checkpoint=args.checkpoint, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    stream = None
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("w", encoding="utf-8")
    try:
        for record in records:
            result = {**record, **score_trace(model, tokenizer, record["question"], record["response"],
                       args.probe_chunk_size, args.tail_tokens, args.prompt_format)}
            text = json.dumps(result, ensure_ascii=False)
            if stream is not None:
                stream.write(text + "\n")
            else:
                print(text)
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
