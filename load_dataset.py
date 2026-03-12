"""
load_dataset.py — Load and format the medical QA dataset using Qwen3's chat template.
"""

from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer
import config


def get_tokenizer() -> AutoTokenizer:
    model_ref = config.MODEL_PATH or config.MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_raw_dataset():
    """Download and optionally cap the medical QA dataset."""
    dataset = hf_load_dataset(
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        trust_remote_code=True,
    )

    if config.MAX_SAMPLES is not None:
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(config.MAX_SAMPLES, len(dataset[split]))))

    return dataset


def format_data(examples, tokenizer: AutoTokenizer) -> dict:
    """Apply Qwen3 chat template to (question, answer) pairs."""
    return {
        "text": [
            tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": a},
                ],
                tokenize=False,
            )
            for q, a in zip(examples["question"], examples["answer"])
        ]
    }


def get_processed_dataset(tokenizer: AutoTokenizer = None):
    """Return train/test splits with formatted text column."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    raw = load_raw_dataset()
    processed = raw.map(
        lambda ex: format_data(ex, tokenizer),
        batched=True,
        remove_columns=raw[config.DATASET_SPLIT].column_names,
    )

    print(f"[load_dataset] Train samples : {len(processed[config.DATASET_SPLIT])}")
    if config.DATASET_TEST_SPLIT in processed:
        print(f"[load_dataset] Test  samples : {len(processed[config.DATASET_TEST_SPLIT])}")

    # Sanity-check one example
    print("\n[load_dataset] Sample formatted text:\n")
    print(processed[config.DATASET_SPLIT][0]["text"][:500], "…\n")

    return processed


if __name__ == "__main__":
    ds = get_processed_dataset()
