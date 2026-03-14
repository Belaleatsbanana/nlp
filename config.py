import torch

# ── Model
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   
MODEL_PATH = None                      
                                       
OUTPUT_DIR = "./qwen-medical"
ADAPTER_DIR = "./qwen-medical-adapter" 

# Dataset 
KAGGLE_DATASET_HANDLE = "pythonafroz/medquad-medical-question-answer-for-ai-research"
KAGGLE_CSV_FILENAME   = "medquad.csv"
DATASET_SPLIT         = "train"
DATASET_TEST_SPLIT    = "test"
TEST_SIZE             = 0.1
RANDOM_SEED           = 42
MAX_SAMPLES           = None   

# Column names
DATASET_QUESTION_COL  = "question"
DATASET_ANSWER_COL    = "answer"
DATASET_SOURCE_COL    = "source"
DATASET_FOCUS_COL     = "focus_area"

#  4-bit Quantisation
BNB_LOAD_IN_4BIT         = True
BNB_4BIT_QUANT_TYPE      = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE   = torch.bfloat16

# LoRA
LORA_R            = 16
LORA_ALPHA        = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
LORA_DROPOUT      = 0.05
LORA_TASK_TYPE    = "CAUSAL_LM"

# Training
TRAIN_BATCH_SIZE            = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE               = 2e-4
MAX_STEPS                   = 300
FP16                        = False
BF16                        = True
OPTIM                       = "paged_adamw_8bit"
LOGGING_STEPS               = 10
NEFTUNE_NOISE_ALPHA         = 5
GRADIENT_CHECKPOINTING      = True
SAVE_STEPS                  = 100
WARMUP_STEPS                = 9

# Inference / Generation
TEMPERATURE     = 0.3
TOP_P           = 0.8
TOP_K           = 20
MAX_NEW_TOKENS  = 2048
NUM_BEAMS       = 1
ENABLE_THINKING = False

# Evaluation
EVAL_BATCH_SIZE       = 8
BERTSCORE_MODEL_TYPE  = "microsoft/deberta-xlarge-mnli"
EVAL_OUTPUT_FILE      = "eval_results.json"
CLINICAL_GUIDELINE_KEYWORDS = [
    "diagnosis", "treatment", "management", "contraindicated",
    "first-line", "second-line", "dosage", "prognosis",
    "complication", "risk factor", "pathophysiology", "etiology",
]
