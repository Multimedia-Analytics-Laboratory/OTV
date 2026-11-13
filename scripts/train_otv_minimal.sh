#!/usr/bin/env bash
set -euo pipefail

# Minimal OTV training launcher.
#
# Required inputs:
#   MODEL_PATH: base reasoner path or HF model id
#   TRACE_PATH: JSONL with fields question, response, correct
#   OUTPUT_DIR: output checkpoint directory
#
# Optional inputs:
#   TOKENIZED_PATH: cached tokenized dataset path
#   PROJECT_DIR: repo root, defaults to current working directory
#   TEMPLATE: LLaMA-Factory chat template, defaults to qwen3
#   VERIFIER_MODE: kv_cache or hidden_state
#   TARGET_MODE: linear

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
TRACE_PATH="${TRACE_PATH:?TRACE_PATH must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
TOKENIZED_PATH="${TOKENIZED_PATH:-${OUTPUT_DIR}/tokenized}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEMPLATE="${TEMPLATE:-qwen3}"
CUTOFF_LEN="${CUTOFF_LEN:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
VERIFIER_MODE="${VERIFIER_MODE:-kv_cache}"
TARGET_MODE="${TARGET_MODE:-linear}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

LORA_TARGET="${LORA_TARGET:-q_proj,k_proj,v_proj}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/apply_otv_patch.py --check
mkdir -p "${OUTPUT_DIR}" "$(dirname "${TOKENIZED_PATH}")"

prepare_cmd=(
  "${PYTHON_BIN}" scripts/otv_prepare_dataset.py
  --model_name_or_path "${MODEL_PATH}"
  --template "${TEMPLATE}"
  --input_path "${TRACE_PATH}"
  --output_path "${TOKENIZED_PATH}"
  --cutoff_len "${CUTOFF_LEN}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  prepare_cmd+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[prepare] ${TRACE_PATH} -> ${TOKENIZED_PATH}"
"${prepare_cmd[@]}"

echo "[train] verifier_mode=${VERIFIER_MODE} target_mode=${TARGET_MODE} output_dir=${OUTPUT_DIR}"
OTV_ENABLE=1 \
OTV_TARGET_MODE="${TARGET_MODE}" \
OTV_VERIFIER_MODE="${VERIFIER_MODE}" \
OTV_FREEZE_RESIZED_VOCAB=true \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
"${PYTHON_BIN}" -m llamafactory.cli train \
  --stage sft \
  --do_train true \
  --dataset otv_minimal \
  --model_name_or_path "${MODEL_PATH}" \
  --template "${TEMPLATE}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --finetuning_type lora \
  --lora_target "${LORA_TARGET}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --plot_loss false \
  --bf16 true \
  --report_to none \
  --disable_gradient_checkpointing true \
  --add_special_tokens "<token of truth>" \
  --resize_vocab true
