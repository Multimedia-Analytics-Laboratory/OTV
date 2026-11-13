"""Numerical and end-to-end checks using random local models, without downloads."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
    Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
)

from otv import FrozenCache, OTVVerifier, make_targets
from otv.train import backward_batch, train_step


torch.set_num_threads(1)


def tiny_model(kind="qwen3"):
    torch.manual_seed(13)
    config_type, model_type = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[kind]
    config = config_type(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, attention_dropout=0.0,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return model_type(config)


def verifier(base=None):
    model = OTVVerifier(base if base is not None else tiny_model(), rank=2, alpha=4,
                        head_dims=(16, 8), head_dropout=0)
    # Nonzero adapter updates make this a check of gated LoRA as well as masking.
    with torch.no_grad():
        for name, parameter in model.lm.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.05)
    return model


@pytest.mark.parametrize("kind", ["qwen3", "qwen2", "llama"])
def test_parallel_matches_independently_prefilled_single_probes(kind):
    model = verifier(tiny_model(kind)).eval()
    ids = torch.tensor([[1, 4, 7, 8, 9, 2]])
    with torch.no_grad():
        cache = model.prefill(ids)
        original = [(k.clone(), v.clone()) for k, v in cache.layers]
        parallel = model(cache)
        independent = []
        for length in range(ids.shape[1] + 1):
            if length:
                prefix = model.prefill(ids[:, :length])
            else:
                prefix = FrozenCache(tuple((k[:, :, :0], v[:, :, :0]) for k, v in cache.layers),
                                     cache.attention_mask[:, :0])
            independent.append(model(prefix, [length]))
        torch.testing.assert_close(parallel, torch.cat(independent, dim=1), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(model(cache, [5, 2, 6]), parallel[:, [5, 2, 6]], atol=2e-6, rtol=1e-5)
        for (k, v), (old_k, old_v) in zip(cache.layers, original):
            assert torch.equal(k, old_k) and torch.equal(v, old_v)
            assert not k.requires_grad and not v.requires_grad


def test_future_tokens_and_padding_do_not_leak_into_a_prefix():
    model = verifier().eval()
    ids = torch.tensor([[1, 4, 7, 8, 9, 2], [1, 4, 7, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        scores = model(model.prefill(ids, mask))
        single = model(model.prefill(ids[1:2, :4]))
        torch.testing.assert_close(scores[1, :5], single[0], atol=2e-6, rtol=1e-5)
        changed = ids[:1].clone()
        changed[:, 4:] = torch.tensor([[11, 12]])
        before = model(model.prefill(ids[:1]), [3])
        after = model(model.prefill(changed), [3])
        torch.testing.assert_close(before, after, atol=2e-6, rtol=1e-5)


def test_chunked_gradient_matches_dense_gradient_and_keeps_reasoner_frozen():
    first = verifier()
    second = deepcopy(first)
    batch = {"input_ids": torch.tensor([[1, 4, 7, 8, 9, 2], [1, 4, 7, 2, 0, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
             "response_starts": torch.tensor([2, 2]), "correct": torch.tensor([1, 0])}
    dense = backward_batch(first, batch, chunk_size=64)
    chunked = backward_batch(second, batch, chunk_size=2)
    assert dense == pytest.approx(chunked, abs=1e-7)
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
        else:
            assert p.grad is None and q.grad is None
    assert first.truth_embedding.grad.abs().sum() > 0
    assert any(p.grad.abs().sum() > 0 for n, p in first.named_parameters() if "lora_B" in n)
    frozen = {n: p.detach().clone() for n, p in first.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in first.parameters() if p.requires_grad), lr=.01)
    assert 0 <= train_step(first, batch, optimizer, chunk_size=2) <= 1
    for name, p in first.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name])


@pytest.mark.parametrize("mode", ["linear", "constant", "sigmoid", "exponential", "wave", "step", "sine"])
def test_outcome_targets_are_bounded_and_valid_response_boundaries_are_selected(mode):
    targets, valid = make_targets(torch.tensor([5, 4]), torch.tensor([2, 1]),
                                  torch.tensor([1, 0]), torch.arange(6), mode)
    assert ((targets >= 0) & (targets <= 1)).all()
    assert valid.sum().item() == 8
    if mode not in ("constant", "sigmoid"):
        assert targets[0, 2] == .5 and targets[0, 5] == 1
        assert targets[1, 1] == .5 and targets[1, 4] == 0


def local_base(directory):
    base = directory / "base"
    tiny_model().save_pretrained(base)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             "Question": 4, ":": 5, "Response": 6, "What": 7, "is": 8,
             "1": 9, "+": 10, "?": 11, "2": 12, "3": 13, "=": 14, ".": 15}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", bos_token="[BOS]",
                                        eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(base)
    return base


def test_save_reload_and_cli_training_scoring(tmp_path):
    base = local_base(tmp_path)
    model = OTVVerifier.from_pretrained(base, head_dims=(16, 8), head_dropout=0, rank=2, alpha=4).eval()
    ids = torch.tensor([[1, 4, 7, 2]])
    with torch.no_grad():
        before = model(model.prefill(ids))
    model.save_pretrained(tmp_path / "roundtrip")
    reloaded = OTVVerifier.from_pretrained(base, checkpoint=tmp_path / "roundtrip").eval()
    with torch.no_grad():
        torch.testing.assert_close(before, reloaded(reloaded.prefill(ids)), rtol=0, atol=1e-7)
    data = tmp_path / "traces.jsonl"
    data.write_text('\n'.join(json.dumps({"question": "What is 1 + 1?", "response": response, "correct": label})
                              for response, label in [("2.", 1), ("3.", 0)]) + '\n')
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    output = tmp_path / "trained"
    common = [sys.executable, "-m"]
    train = subprocess.run(common + ["otv.train", "--model", str(base), "--train-data", str(data),
                           "--eval-data", str(data),
                           "--output", str(output), "--epochs", "1", "--batch-size", "2",
                           "--gradient-accumulation", "1", "--head-dims", "16,8", "--head-dropout", "0",
                           "--probe-chunk-size", "3", "--save-steps", "0", "--max-length", "64"],
                           env=env, capture_output=True, text=True, timeout=90)
    assert train.returncode == 0, train.stdout + train.stderr
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1 and 0 <= metrics[0]["eval_mse"] <= 1
    scores = tmp_path / "scores.jsonl"
    score = subprocess.run(common + ["otv.score", "--checkpoint", str(output), "--input", str(data),
                           "--output", str(scores), "--probe-chunk-size", "2"],
                           env=env, capture_output=True, text=True, timeout=90)
    assert score.returncode == 0, score.stdout + score.stderr
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    assert len(rows) == 2 and all(0 <= row["trace_score"] <= 1 for row in rows)
    assert all(len(row["prefix_scores"]) == len(row["response_tokens_seen"]) for row in rows)
