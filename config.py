"""
config.py — Central configuration for Qwen2.5 medical fine-tuning pipeline.

Precision Tweaks applied:
  - per_device_train_batch_size lowered to 1 with gradient_accumulation_steps=8
    for better gradient stability on small GPU memory.
  - temperature lowered to 0.3 (from 0.7) for stricter factual adherence
    in medical inference.
"""

import torch

# ── Model
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # HuggingFace hub ID (fallback)
MODEL_PATH = None                      # Set to local path or Kaggle download path;
                                       # if None, MODEL_ID is used.
OUTPUT_DIR = "./qwen-medical"
ADAPTER_DIR = "./qwen-medical-adapter" # Where the LoRA weights are saved after training

# ── Dataset ────────────────────────────────────────────────────────────────────
# MedQuAD downloaded directly from Kaggle via kagglehub → pandas DataFrame.
KAGGLE_DATASET_HANDLE = "pythonafroz/medquad-medical-question-answer-for-ai-research"
KAGGLE_CSV_FILENAME   = "medquad.csv"
DATASET_SPLIT         = "train"
DATASET_TEST_SPLIT    = "test"
TEST_SIZE             = 0.1
RANDOM_SEED           = 42
MAX_SAMPLES           = None   # int to cap rows during debugging

# Column names (MedQuAD schema)
DATASET_QUESTION_COL  = "question"
DATASET_ANSWER_COL    = "answer"
DATASET_SOURCE_COL    = "source"
DATASET_FOCUS_COL     = "focus_area"

# ── 4-bit Quantisation
BNB_LOAD_IN_4BIT         = True
BNB_4BIT_QUANT_TYPE      = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE   = torch.bfloat16

# ── LoRA────
LORA_R            = 16
LORA_ALPHA        = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
LORA_DROPOUT      = 0.05
LORA_TASK_TYPE    = "CAUSAL_LM"

# ── Training
# Precision Tweak: batch_size=1 + accum=8 gives effective batch of 8 while
# keeping per-step gradient variance low, which helps with medical factuality.
TRAIN_BATCH_SIZE            = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE               = 2e-4
MAX_STEPS                   = 300
FP16                        = False  # Must be False when compute dtype is bfloat16
BF16                        = True   # bfloat16 has wider exponent range; no grad scaler needed
OPTIM                       = "paged_adamw_8bit"  # adamw_8bit can still trigger grad scaler with some accelerate versions
LOGGING_STEPS               = 10
NEFTUNE_NOISE_ALPHA         = 5
GRADIENT_CHECKPOINTING      = True
SAVE_STEPS                  = 100
WARMUP_STEPS                = 9     # ~3% of MAX_STEPS (replaces deprecated warmup_ratio)

# ── Inference / Generation ─────────────────────────────────────────────────────
# Precision Tweak: temperature=0.3 reduces hallucination for factual medical Q&A.
TEMPERATURE     = 0.3
TOP_P           = 0.8
TOP_K           = 20
MAX_NEW_TOKENS  = 2048
NUM_BEAMS       = 1
ENABLE_THINKING = False  # Qwen2.5 does not support chain-of-thought thinking tokens

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE       = 8
BERTSCORE_MODEL_TYPE  = "microsoft/deberta-xlarge-mnli"  # Best BERTScore backbone
EVAL_OUTPUT_FILE      = "eval_results.json"
# Keywords drawn from UpToDate® clinical guidance areas used for keyword-match
# medical accuracy heuristics (extend as needed).
CLINICAL_GUIDELINE_KEYWORDS = [
    "diagnosis", "treatment", "management", "contraindicated",
    "first-line", "second-line", "dosage", "prognosis",
    "complication", "risk factor", "pathophysiology", "etiology",
]
