"""
train.py — Fine-tune Qwen2.5-1.5B-Instruct using either QLoRA or DoRA.

Run this script and it will ask you which method to use.
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


def train_qlora(model, tokenizer, train_dataset):
    print("[train] Using QLoRA...")
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        task_type=config.LORA_TASK_TYPE,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=build_training_args(),
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=lambda x: x["text"],
    )
    trainer.train()
    save_dir = config.ADAPTER_DIR.rstrip("/") + "-qlora"
    print(f"[train] Saving QLoRA adapter to {save_dir}")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def train_dora(model, tokenizer, train_dataset):
    print("[train] Using DoRA...")
    dora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        task_type=config.LORA_TASK_TYPE,
        use_dora=True,
    )
    model = get_peft_model(model, dora_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=build_training_args(),
        train_dataset=train_dataset,
        formatting_func=lambda x: x["text"],
    )
    trainer.train()
    save_dir = config.ADAPTER_DIR.rstrip("/") + "-dora"
    print(f"[train] Saving DoRA adapter to {save_dir}")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():
    print("Choose fine-tuning method:")
    print("1. QLoRA")
    print("2. DoRA")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in ("1", "2"):
        print("Invalid choice. Exiting.")
        return

    print("[train] Loading model and tokenizer (4-bit)...")
    model, tokenizer = load_model_and_tokenizer()

    print("[train] Loading and formatting dataset...")
    dataset = get_processed_dataset(tokenizer)
    train_dataset = dataset[config.DATASET_SPLIT]

    if choice == "1":
        train_qlora(model, tokenizer, train_dataset)
    else:
        train_dora(model, tokenizer, train_dataset)

    print("[train] Done.")


if __name__ == "__main__":
    main()