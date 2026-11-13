"""JSONL trace loading and deterministic prompt/response token boundaries."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        if str(path).endswith(".jsonl"):
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Input must contain a nonempty list of traces.")
    return records


def encode_trace(tokenizer, question, response, max_length=None, prompt_format="auto"):
    if not isinstance(question, str) or not isinstance(response, str) or not response.strip():
        raise ValueError("question and a nonempty response must be strings.")
    if prompt_format not in ("auto", "chat", "plain"):
        raise ValueError("prompt_format must be auto, chat, or plain.")
    use_chat = prompt_format == "chat" or (prompt_format == "auto" and tokenizer.chat_template)
    if use_chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
    else:
        prompt = tokenizer.encode(f"Question: {question}\nResponse: ", add_special_tokens=True)
    answer = tokenizer.encode(response, add_special_tokens=False)
    if not answer:
        raise ValueError("The response contains no tokens.")
    if tokenizer.eos_token_id is not None and answer[-1] != tokenizer.eos_token_id:
        answer.append(tokenizer.eos_token_id)
    ids = prompt + answer
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f"A trace has {len(ids)} tokens, exceeding max_length={max_length}; "
            "increase the limit or filter the data. Completed traces are not silently truncated."
        )
    return {"input_ids": ids, "response_start": len(prompt)}


class TraceDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=4096, prompt_format="auto"):
        self.rows = []
        for index, row in enumerate(load_records(path)):
            label = row.get("correct")
            if label not in (0, 1):
                raise ValueError(f"Trace {index}: correct must be 0 or 1.")
            encoded = encode_trace(tokenizer, row["question"], row["response"], max_length, prompt_format)
            self.rows.append({**encoded, "correct": int(label)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TraceCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            size = len(row["input_ids"])
            ids[index, :size] = torch.tensor(row["input_ids"])
            mask[index, :size] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "response_starts": torch.tensor([row["response_start"] for row in rows]),
                "correct": torch.tensor([row["correct"] for row in rows])}
