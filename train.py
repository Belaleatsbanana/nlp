"""
train.py — Fine-tune Qwen2.5-1.5B-Instruct using DoRA (Weight-Decomposed Low-Rank Adaptation).

DoRA decomposes weights into magnitude and direction, learning a separate magnitude vector
while adapting the direction via low-rank updates. This often outperforms LoRA with the same
parameter budget. Here we use PEFT's built-in DoRA support (use_dora=True in LoraConfig).
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import config
from load_dataset import get_processed_dataset


# ---------------------------------------------------------------------------
# Model loading (4-bit quantised for memory efficiency)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    )
    model_ref = config.MODEL_PATH or config.MODEL_ID

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA / DoRA configuration
# ---------------------------------------------------------------------------

def get_dora_config():
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        task_type=config.LORA_TASK_TYPE,
        use_dora=True,  # <-- the only change from standard LoRA
    )


# ---------------------------------------------------------------------------
# Training arguments (same as before, but no LISA callback)
# ---------------------------------------------------------------------------

def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        max_steps=config.MAX_STEPS,
        fp16=config.FP16,
        bf16=config.BF16,
        optim=config.OPTIM,
        report_to="none",
        logging_steps=config.LOGGING_STEPS,
        neftune_noise_alpha=config.NEFTUNE_NOISE_ALPHA,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        save_strategy="no",                 # no intermediate checkpoints
        warmup_steps=config.WARMUP_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,
    )


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train():
    print("[train] Loading model and tokenizer (4-bit)...")
    model, tokenizer = load_model_and_tokenizer()

    print("[train] Applying DoRA configuration...")
    dora_config = get_dora_config()
    model = get_peft_model(model, dora_config)
    model.print_trainable_parameters()  # shows how many parameters are trainable

    print("[train] Loading and formatting dataset...")
    dataset = get_processed_dataset(tokenizer)
    train_dataset = dataset[config.DATASET_SPLIT]

    training_args = build_training_args()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=lambda x: x["text"],
    )

    print("[train] Starting DoRA training...")
    trainer.train()

    # Save the final adapter (DoRA) to a distinct directory
    dora_adapter_dir = config.ADAPTER_DIR.rstrip("/") + "-dora"
    print(f"[train] Saving model to {dora_adapter_dir}")
    trainer.model.save_pretrained(dora_adapter_dir)
    tokenizer.save_pretrained(dora_adapter_dir)
    print("[train] Done.")


if __name__ == "__main__":
    train()