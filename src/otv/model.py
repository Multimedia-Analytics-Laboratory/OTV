"""Query frozen reasoning caches without modifying the installed model library."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class FrozenCache:
    """Detached full-attention K/V states; verification never appends to them."""

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor

    @property
    def lengths(self):
        return self.attention_mask.sum(-1).long()

    @property
    def sequence_length(self):
        return self.attention_mask.shape[1]

    def working_copy(self):
        # DynamicLayer.update concatenates into a new tensor. Its container is
        # private to this call, so the detached original tensors remain intact.
        cache = DynamicCache()
        for index, (keys, values) in enumerate(self.layers):
            cache.update(keys, values, index)
        return cache


def probe_mask(attention_mask, positions, dtype):
    """Each query reads its cached prefix and only its own new K/V pair."""
    batch, length = attention_mask.shape
    queries = positions.shape[1]
    cached = torch.arange(length, device=positions.device)[None, None, :]
    cached = (cached < positions[:, :, None]) & attention_mask[:, None, :].bool()
    own_probe = torch.eye(queries, device=positions.device, dtype=torch.bool)
    allowed = torch.cat([cached, own_probe.expand(batch, -1, -1)], dim=-1)
    mask = torch.zeros(allowed.shape, device=positions.device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)


class RegressionHead(nn.Module):
    def __init__(self, hidden_size, hidden_dims=(1024, 256), dropout=0.2):
        super().__init__()
        if not hidden_dims or any(int(size) <= 0 for size in hidden_dims):
            raise ValueError("Head hidden dimensions must be positive.")
        blocks = []
        previous = hidden_size
        for size in hidden_dims:
            blocks.extend([nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)])
            previous = size
        blocks.extend([nn.Linear(previous, 1), nn.Sigmoid()])
        self.layers = nn.Sequential(*blocks)

    def forward(self, hidden):
        return self.layers(hidden.to(self.layers[0].weight.dtype)).squeeze(-1)


class OTVVerifier(nn.Module):
    """Single-device reference implementation for full-attention Qwen3/Qwen2/Llama.

    A frozen, adapter-disabled prefill produces the reasoning cache. A standard
    model forward then processes learned truth embeddings with adapters enabled,
    explicit prefix positions, and a cache-plus-diagonal additive attention mask.
    No attention classes or site-packages files are replaced.
    """

    FORMAT_VERSION = 1
    MODEL_TYPES = {"qwen3", "qwen2", "llama"}

    def __init__(self, base_model, rank=8, alpha=16, lora_dropout=0.0,
                 head_dims=(1024, 256), head_dropout=0.2):
        super().__init__()
        config = base_model.config
        if config.model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported architecture: {config.model_type}.")
        if (getattr(config, "use_sliding_window", False)
                or "sliding_attention" in (getattr(config, "layer_types", None) or [])):
            raise ValueError("This implementation requires full-attention layers.")
        if isinstance(base_model, PeftModel):
            self.lm = base_model
        else:
            base_model.set_attn_implementation("eager")
            self.lm = get_peft_model(base_model, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank, lora_alpha=alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none",
            ))
        self.lm.get_base_model().set_attn_implementation("eager")
        self.head_dims = list(head_dims)
        self.head_dropout = head_dropout
        device = self.lm.get_input_embeddings().weight.device
        self.truth_embedding = nn.Parameter(torch.empty(
            config.hidden_size, device=device, dtype=torch.float32
        ).normal_(std=config.initializer_range))
        self.head = RegressionHead(config.hidden_size, head_dims, head_dropout).to(device)

    @property
    def decoder(self):
        return self.lm.get_base_model().model

    @property
    def device(self):
        return self.truth_embedding.device

    @torch.no_grad()
    def prefill(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids.")
        lengths = attention_mask.sum(-1).long()
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        if (lengths == 0).any() or not torch.equal(attention_mask.bool(), expected):
            raise ValueError("Use nonempty sequences with contiguous right padding.")
        was_training = self.lm.training
        try:
            self.lm.eval()
            with self.lm.disable_adapter():
                output = self.decoder(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True,
                )
            layers = tuple((layer.keys.detach(), layer.values.detach())
                           for layer in output.past_key_values.layers)
            if any(keys.shape[-2] != input_ids.shape[1] for keys, _ in layers):
                raise ValueError("The model truncated its cache; full attention is required.")
            return FrozenCache(layers, attention_mask.detach().clone())
        finally:
            self.lm.train(was_training)

    def forward(self, cache, prefix_lengths=None):
        if prefix_lengths is None:
            prefix_lengths = torch.arange(cache.sequence_length + 1, device=self.device)
            prefix_lengths = prefix_lengths[None].expand(cache.lengths.shape[0], -1)
            prefix_lengths = torch.minimum(prefix_lengths, cache.lengths[:, None])
        else:
            prefix_lengths = torch.as_tensor(prefix_lengths, device=self.device)
            if prefix_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("Prefix lengths must be integers.")
            if prefix_lengths.ndim == 1:
                prefix_lengths = prefix_lengths[None].expand(cache.lengths.shape[0], -1)
        if (prefix_lengths.ndim != 2 or prefix_lengths.shape[0] != cache.lengths.shape[0]
                or prefix_lengths.shape[1] == 0):
            raise ValueError("Prefix lengths must have shape [queries] or [batch, queries].")
        if ((prefix_lengths < 0) | (prefix_lengths > cache.lengths[:, None])).any():
            raise ValueError("Each prefix must be within its unpadded cached sequence.")
        positions = prefix_lengths.long()
        dtype = self.lm.get_input_embeddings().weight.dtype
        probes = self.truth_embedding.to(dtype)[None, None, :].expand(
            positions.shape[0], positions.shape[1], -1
        )
        output = self.decoder(
            inputs_embeds=probes,
            attention_mask=probe_mask(cache.attention_mask, positions, dtype),
            past_key_values=cache.working_copy(),
            position_ids=positions,
            cache_position=torch.arange(cache.sequence_length,
                                        cache.sequence_length + positions.shape[1],
                                        device=self.device),
            use_cache=True, return_dict=True,
        )
        return self.head(output.last_hidden_state)

    def save_pretrained(self, directory, tokenizer=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
        state = {"truth_embedding": self.truth_embedding.detach().cpu().contiguous()}
        state.update({f"head.{k}": v.detach().cpu().contiguous()
                      for k, v in self.head.state_dict().items()})
        save_file(state, str(directory / "otv_head.safetensors"))
        config = {
            "format_version": self.FORMAT_VERSION,
            "base_model_name_or_path": self.lm.get_base_model().config._name_or_path,
            "model_type": self.lm.config.model_type,
            "head_dims": self.head_dims, "head_dropout": self.head_dropout,
        }
        (directory / "otv_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path=None, checkpoint=None, device="cpu",
                        dtype=torch.float32, **kwargs):
        metadata = None
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            metadata = json.loads((checkpoint / "otv_config.json").read_text())
            if metadata["format_version"] != cls.FORMAT_VERSION:
                raise ValueError("Unsupported standalone checkpoint format.")
            model_name_or_path = model_name_or_path or metadata["base_model_name_or_path"]
        if not model_name_or_path:
            raise ValueError("Supply the original base model path or Hugging Face ID.")
        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, attn_implementation="eager"
        ).to(device)
        if metadata is None:
            return cls(base, **kwargs)
        if base.config.model_type != metadata["model_type"]:
            raise ValueError("The checkpoint and base model architectures differ.")
        peft = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
        model = cls(peft, head_dims=metadata["head_dims"], head_dropout=metadata["head_dropout"])
        state = load_file(str(checkpoint / "otv_head.safetensors"), device="cpu")
        with torch.no_grad():
            model.truth_embedding.copy_(state.pop("truth_embedding").to(model.device))
        model.head.load_state_dict({k.removeprefix("head."): v for k, v in state.items()}, strict=True)
        return model
