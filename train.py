import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 with QLoRA or DoRA")
    parser.add_argument("--method", type=str, required=True,
                        choices=["qlora", "dora"],
                        help="Fine-tuning method: qlora or dora")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override config.OUTPUT_DIR")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Override config.ADAPTER_DIR (suffix -qlora/-dora will be added)")
    return parser.parse_args()


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
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
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
        save_strategy="no",
        warmup_steps=config.WARMUP_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,
    )


def train_qlora(model, tokenizer, train_dataset, output_dir, adapter_base):
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
        args=build_training_args(output_dir),
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=lambda x: x["text"],
    )
    trainer.train()
    save_dir = adapter_base.rstrip("/") + "-qlora"
    print(f"[train] Saving QLoRA adapter to {save_dir}")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def train_dora(model, tokenizer, train_dataset, output_dir, adapter_base):
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
        args=build_training_args(output_dir),
        train_dataset=train_dataset,
        formatting_func=lambda x: x["text"],
    )
    trainer.train()
    save_dir = adapter_base.rstrip("/") + "-dora"
    print(f"[train] Saving DoRA adapter to {save_dir}")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():
    args = parse_args()

    # Use command-line overrides or fall back to config
    output_dir = args.output_dir if args.output_dir else config.OUTPUT_DIR
    adapter_base = args.adapter_dir if args.adapter_dir else config.ADAPTER_DIR

    print("[train] Loading model and tokenizer (4-bit)...")
    model, tokenizer = load_model_and_tokenizer()

    print("[train] Loading and formatting dataset...")
    dataset = get_processed_dataset(tokenizer)
    train_dataset = dataset[config.DATASET_SPLIT]

    if args.method == "qlora":
        train_qlora(model, tokenizer, train_dataset, output_dir, adapter_base)
    else:
        train_dora(model, tokenizer, train_dataset, output_dir, adapter_base)

    print("[train] Done.")


if __name__ == "__main__":
    main()