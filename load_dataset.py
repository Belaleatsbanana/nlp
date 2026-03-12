"""
load_dataset.py — Download MedQuAD directly from Kaggle via kagglehub,
load into a pandas DataFrame (no read_csv), then convert to a HuggingFace
DatasetDict and apply Qwen2.5's chat template.

Requirements: kagglehub  (pip install kagglehub)
Credentials : set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place
              ~/.kaggle/kaggle.json  ({"username": "...", "key": "..."})
"""

import csv
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import config



# Tokenizer
def get_tokenizer() -> AutoTokenizer:
    model_ref = config.MODEL_PATH or config.MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Kaggle unto pandas DataFrame
def load_raw_df() -> pd.DataFrame:
    """
    Download MedQuAD from Kaggle via kagglehub and load it directly into a
    pandas DataFrame

    Returns columns: question, answer, source, focus_area
    """

    kaggle_dataset_handle = config.KAGGLE_DATASET_HANDLE
    kaggle_csv_filename   = config.KAGGLE_CSV_FILENAME
    


    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        kaggle_dataset_handle,
        kaggle_csv_filename,
    )

    # Rename 34an kaggle by3abat
    original_cols = list(df.columns)
    rename_map = {
        original_cols[0]: config.DATASET_QUESTION_COL,
        original_cols[1]: config.DATASET_ANSWER_COL,
        original_cols[2]: config.DATASET_SOURCE_COL,
        original_cols[3]: config.DATASET_FOCUS_COL,
    }
    df = df.rename(columns=rename_map)

    # Force question and answer to str, drop any rows where they are null/empty
    for col in [config.DATASET_QUESTION_COL, config.DATASET_ANSWER_COL]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[
        df[config.DATASET_QUESTION_COL].ne("") &
        df[config.DATASET_ANSWER_COL].ne("")
    ].reset_index(drop=True)

    if (dropped := before - len(df)):
        print(f"[load_dataset] Dropped {dropped} rows with missing Q/A.")

    if config.MAX_SAMPLES is not None:
        df = df.head(config.MAX_SAMPLES)

    print(f"[load_dataset] Rows        : {len(df)}")

    return df


# pandas to HuggingFace DatasetDict
def df_to_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    """Split the DataFrame 90/10 and wrap as a HuggingFace DatasetDict."""
    hf_ds = Dataset.from_pandas(df, preserve_index=False)
    split = hf_ds.train_test_split(test_size=config.TEST_SIZE, seed=config.RANDOM_SEED)
    return DatasetDict({
        config.DATASET_SPLIT:      split["train"],
        config.DATASET_TEST_SPLIT: split["test"],
    })


# Chat template formatting

def format_data(examples, tokenizer: AutoTokenizer) -> dict:
    return {
        "text": [
            tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": str(q)},
                    {"role": "assistant", "content": str(a)},
                ],
                tokenize=False,
                enable_thinking=False,  # Qwen2.5 does not use thinking tokens
            )
            for q, a in zip(
                examples[config.DATASET_QUESTION_COL],
                examples[config.DATASET_ANSWER_COL],
            )
        ]
    }


# entry point

def get_processed_dataset(tokenizer: AutoTokenizer = None) -> DatasetDict:
    """
    Full pipeline:
      Kaggle API -> pandas DataFrame -> HuggingFace DatasetDict -> chat-formatted 'text' column
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    df  = load_raw_df()
    raw = df_to_dataset_dict(df)

    processed = raw.map(
        lambda ex: format_data(ex, tokenizer),
        batched=True,
        remove_columns=raw[config.DATASET_SPLIT].column_names,
    )

    print(f"[load_dataset] Train samples : {len(processed[config.DATASET_SPLIT])}")
    print(f"[load_dataset] Test  samples : {len(processed[config.DATASET_TEST_SPLIT])}")
    print("\n[load_dataset] Sample formatted text:\n")
    print(processed[config.DATASET_SPLIT][0]["text"][:500], "…\n")

    return processed


if __name__ == "__main__":
    df = load_raw_df()
    get_processed_dataset()
