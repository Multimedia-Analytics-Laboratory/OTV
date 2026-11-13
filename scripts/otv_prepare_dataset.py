"""Convert reasoning traces into tokenized OTV training datasets."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {path}.")


def first_response_index(labels: list[int]) -> int:
    for idx, label in enumerate(labels):
        if label != IGNORE_INDEX:
            return idx
    raise ValueError("No response token found after tokenization.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a tokenized OTV dataset from JSON or JSONL traces.")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path or Hugging Face model ID.")
    parser.add_argument("--template", required=True, help="LLaMA-Factory chat template.")
    parser.add_argument("--input_path", required=True, help="Input trace file in JSON or JSONL format.")
    parser.add_argument("--output_path", required=True, help="Output directory for the tokenized dataset.")
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--correct_field", default="correct")
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_size", type=float, default=0.0, help="Validation fraction; zero creates only train.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)

    processed = []
    for record in records:
        prompt = [{"role": "user", "content": record[args.question_field]}]
        response = [{"role": "assistant", "content": record[args.response_field]}]
        input_ids, labels = processor._encode_data_example(
            prompt=prompt,
            response=response,
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
        )
        processed.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "otv_correct": int(record[args.correct_field]),
                "response_start": first_response_index(labels),
            }
        )

    dataset = Dataset.from_list(processed)
    if args.eval_size > 0:
        split = dataset.train_test_split(test_size=args.eval_size, seed=42)
        dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(str(output_path))
    print(f"Saved tokenized OTV dataset to {output_path}")


if __name__ == "__main__":
    main()
