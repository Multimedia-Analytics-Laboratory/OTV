# OTV

Core training and checkpoint evaluation code for OTV. The verifier reuses the reasoner's KV cache to score reasoning prefixes; a hidden-state baseline is also supported.

## Blog

[**OTV Blog Download**](docs/index.html?raw=1)

Download the HTML file and open it in your browser to read the blog and use its interactive figures. Images, plots, styles, scripts, and the paper citation are embedded in this single file.

## Installation

Use Python 3.11 and Git. Create a dedicated environment, install the linked framework, and apply the OTV changes:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/apply_otv_patch.py
```

`requirements.txt` links to the official [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) repository in editable mode, pinned to v0.9.4 commit `95ac3f2373b82662c1bd855c284d3379e6a763d3`. Pip keeps its source checkout inside the environment by default. The framework source is fetched as a dependency rather than copied into OTV.

The patch adds the regression head and OTV training logic in two upstream files: `model/adapter.py` and `train/sft/trainer.py`. The application script verifies the Git commit and source checksums, supports repeat execution, and refuses to overwrite local edits. Verify the installed patch at any time:

```bash
python scripts/apply_otv_patch.py --check
```

The training launcher performs this check before preparing data. Reinstalling the dependency may restore its original files; reapply the patch afterward.

## Data Preparation

Use JSONL with one reasoning trace per line:

```json
{"question": "What is 1 + 1?", "response": "1 + 1 = 2. Final answer: 2.", "correct": 1}
{"question": "What is 1 + 1?", "response": "1 + 1 = 3. Final answer: 3.", "correct": 0}
```

`question` is the prompt, `response` is the full reasoning trace, and `correct` is its final correctness label (`0` or `1`). [Synthetic examples](examples/traces.example.jsonl) illustrate the format. Real labels require an appropriate task-specific validator.

`scripts/otv_generate_traces.py` optionally generates traces from questions and reference answers. Its default input fields are `query` and `response`; select other fields with `--question_field` and `--answer_field`. Its answer normalization targets mathematical answers.

`scripts/otv_prepare_dataset.py` converts traces into tokenized datasets. It creates `train` by default and adds `validation` when `--eval_size` is set.

## Training

Run from the repository root after activating the environment:

```bash
MODEL_PATH=/path/to/base-model \
TRACE_PATH=/path/to/train_traces.jsonl \
OUTPUT_DIR="$(pwd)/outputs/otv_run" \
CUDA_DEVICE=0 \
TEMPLATE=qwen3 \
CUTOFF_LEN=2048 \
NUM_TRAIN_EPOCHS=1 \
bash scripts/train_otv_minimal.sh
```

The launcher prepares data and starts LoRA training with CUDA and bf16. Defaults include `VERIFIER_MODE=kv_cache`, `TARGET_MODE=linear`, `LORA_RANK=8`, batch size `1`, and `8` gradient accumulation steps. Set `VERIFIER_MODE=hidden_state` in the same invocation for the baseline.

## Evaluation

Prepare held-out tokenized data using the same tokenizer and template, then evaluate saved checkpoints:

```bash
python scripts/evaluate_otv_checkpoints.py \
  --base_model /path/to/base-model \
  --run_dir /path/to/otv_run \
  --tokenized_path /path/to/tokenized-evaluation-data \
  --split validation \
  --verifier_mode kv_cache \
  --include_final \
  --output_path /path/to/evaluation-results.json
```

The evaluation dataset must contain the requested split. The minimal training launcher creates only `train`; prepare validation data separately or select another existing split with `--split`. Match `--verifier_mode` to the training run.

## Checkpoints

A complete checkpoint contains:

- `adapter_model.safetensors`: LoRA weights.
- `reg_model.safetensors`: OTV regression head weights.
- `reg_model_config.json`: regression head configuration.

Intermediate checkpoints use `checkpoint-*` directories. Data, model weights, local environments, logs, and generated results are excluded through `.gitignore`.

## Repository Structure

```text
OTV/
├── docs/
│   └── index.html                   # Downloadable interactive OTV blog
├── requirements.txt                 # Link to the pinned upstream dependency
├── patches/
│   ├── llamafactory-otv.patch        # OTV changes to two framework files
│   └── LICENSE                      # License for upstream-derived patch content
├── scripts/
│   ├── apply_otv_patch.py
│   ├── otv_generate_traces.py
│   ├── otv_prepare_dataset.py
│   ├── train_otv_minimal.sh
│   └── evaluate_otv_checkpoints.py
└── examples/
    └── traces.example.jsonl
```

## Acknowledgement

OTV builds on LLaMA-Factory. Upstream source headers are retained, and the upstream Apache-2.0 license is included with the patch. The OTV-modified framework files are reconstructed from the supplied training snapshot.

The patch installation and source checks can be tested independently of model loading. Full training and numerical equivalence have not been verified on this machine; they require the actual model, data, and supported hardware.
