"""
config.py — Central configuration for Qwen3 medical fine-tuning pipeline.

Precision Tweaks applied:
  - per_device_train_batch_size lowered to 1 with gradient_accumulation_steps=8
    for better gradient stability on small GPU memory.
  - temperature lowered to 0.3 (from 0.7) for stricter factual adherence
    in medical inference.
"""

import torch

# ── Model
MODEL_ID = "Qwen/Qwen3-1.7B"          # HuggingFace hub ID (fallback)
MODEL_PATH = None                      # Set to local path or Kaggle download path;
                                       # if None, MODEL_ID is used.
OUTPUT_DIR = "./qwen-medical"
ADAPTER_DIR = "./qwen-medical-adapter" # Where the LoRA weights are saved after training

# ── Dataset─
DATASET_NAME = "lavita/medical-qa-datasets"
DATASET_CONFIG = "chatdoctor_healthcaremagic"  # subset with question/answer columns
DATASET_SPLIT = "train"
DATASET_TEST_SPLIT = "test"
MAX_SAMPLES = None       # Set an int to cap dataset size during debugging

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
FP16                        = True
OPTIM                       = "adamw_8bit"
LOGGING_STEPS               = 10
NEFTUNE_NOISE_ALPHA         = 5
GRADIENT_CHECKPOINTING      = True
SAVE_STEPS                  = 100
WARMUP_RATIO                = 0.03

# ── Inference / Generation ─────────────────────────────────────────────────────
# Precision Tweak: temperature=0.3 reduces hallucination for factual medical Q&A.
TEMPERATURE     = 0.3
TOP_P           = 0.8
TOP_K           = 20
MAX_NEW_TOKENS  = 2048
NUM_BEAMS       = 1
ENABLE_THINKING = True   # Qwen3 chain-of-thought flag

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
