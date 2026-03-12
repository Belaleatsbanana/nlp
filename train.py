"""
train.py — Fine-tune Qwen2.5-1.5B-Instruct on a medical QA dataset using QLoRA (4-bit + LoRA).

Precision Tweaks:
  - per_device_train_batch_size=1 with gradient_accumulation_steps=8 stabilises
    gradients without requiring more VRAM.
  - NEFTune noise (alpha=5) improves instruction-following robustness.
  - warmup_ratio=0.03 prevents early training instability.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

import config
from load_dataset import get_tokenizer, get_processed_dataset


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    )


def load_model_and_tokenizer():
    model_ref = config.MODEL_PATH or config.MODEL_ID
    bnb_config = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        task_type=config.LORA_TASK_TYPE,
    )


def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        # Precision Tweak: small batch + high accumulation → stable gradients
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        max_steps=config.MAX_STEPS,
        fp16=config.FP16,   # False — bfloat16 compute dtype is incompatible with fp16 scaler
        bf16=config.BF16,   # True  — bfloat16 needs no grad scaler, avoids unscale_ error
        optim=config.OPTIM,
        report_to="none",
        logging_steps=config.LOGGING_STEPS,
        neftune_noise_alpha=config.NEFTUNE_NOISE_ALPHA,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        save_steps=config.SAVE_STEPS,
        warmup_steps=config.WARMUP_STEPS,  # replaces deprecated warmup_ratio
        save_total_limit=2,
        load_best_model_at_end=False,
    )


def train():
    print("[train] Loading model and tokenizer…")
    model, tokenizer = load_model_and_tokenizer()

    print("[train] Loading and formatting dataset…")
    dataset = get_processed_dataset(tokenizer)
    train_dataset = dataset[config.DATASET_SPLIT]

    peft_config   = build_lora_config()
    training_args = build_training_args()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=lambda x: x["text"],
    )

    print("[train] Starting training…")
    trainer.train()

    print(f"[train] Saving adapter to {config.ADAPTER_DIR}")
    trainer.model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)
    print("[train] Done.")


if __name__ == "__main__":
    train()
