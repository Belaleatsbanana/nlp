"""
eval.py — Evaluate the fine-tuned Qwen2.5 medical model.

Metrics:
  1. BERTScore (Precision / Recall / F1) using DeBERTa-xlarge-mnli backbone.
  2. Medical Accuracy Score — keyword-overlap heuristic against a set of
     UpToDate®-inspired clinical guideline terms (see config.CLINICAL_GUIDELINE_KEYWORDS).
     Acts as a lightweight proxy until a full clinical NLI model is integrated.
  3. ROUGE-L for surface-level fluency reference.
"""

import json
import torch
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from bert_score import score as bertscore
from rouge_score import rouge_scorer

import config


# ── Model loading ──────────────────────────────────────────────────────────────

def load_eval_model():
    """Load quantised base model with LoRA adapter merged for inference."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    )
    model_ref = config.MODEL_PATH or config.MODEL_ID

    base_model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, question: str) -> str:
    """Generate a model answer for a single question."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,   # Precision Tweak: 0.3 for factual adherence
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            num_beams=config.NUM_BEAMS,
            do_sample=True,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Medical accuracy heuristic ─────────────────────────────────────────────────

def medical_accuracy_score(prediction: str, reference: str) -> float:
    """
    Keyword-overlap proxy for clinical guideline adherence.

    Checks what fraction of UpToDate®-inspired clinical keywords present in the
    *reference* answer also appear in the *prediction*. Returns a score in [0, 1].
    """
    ref_lower  = reference.lower()
    pred_lower = prediction.lower()

    relevant_keywords = [
        kw for kw in config.CLINICAL_GUIDELINE_KEYWORDS if kw in ref_lower
    ]
    if not relevant_keywords:
        return 1.0  # No clinical keywords in reference → not penalised

    matched = sum(1 for kw in relevant_keywords if kw in pred_lower)
    return matched / len(relevant_keywords)


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return {"rougeL_mean": sum(scores) / len(scores)}


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(num_samples: int = 100):
    print("[eval] Loading model…")
    model, tokenizer = load_eval_model()

    print("[eval] Loading test set from CSV…")
    from load_dataset import load_raw_df, df_to_dataset_dict
    df  = load_raw_df()
    dsd = df_to_dataset_dict(df)
    raw = dsd[config.DATASET_TEST_SPLIT]
    if num_samples:
        raw = raw.select(range(min(num_samples, len(raw))))

    questions  = raw[config.DATASET_QUESTION_COL]
    references = raw[config.DATASET_ANSWER_COL]

    predictions   = []
    med_acc_scores = []

    print(f"[eval] Generating answers for {len(questions)} samples…")
    for q, ref in tqdm(zip(questions, references), total=len(questions)):
        pred = generate_answer(model, tokenizer, q)
        predictions.append(pred)
        med_acc_scores.append(medical_accuracy_score(pred, ref))

    # ── BERTScore ──────────────────────────────────────────────────────────────
    print("[eval] Computing BERTScore…")
    P, R, F1 = bertscore(
        predictions,
        references,
        model_type=config.BERTSCORE_MODEL_TYPE,
        lang="en",
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_results = {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item(),
    }

    # ── ROUGE-L ────────────────────────────────────────────────────────────────
    print("[eval] Computing ROUGE-L…")
    rouge_results = compute_rouge(predictions, references)

    # ── Medical accuracy ───────────────────────────────────────────────────────
    med_acc_mean = sum(med_acc_scores) / len(med_acc_scores)

    results = {
        **bert_results,
        **rouge_results,
        "medical_accuracy_keyword_mean": med_acc_mean,
        "num_samples": len(questions),
    }

    print("\n[eval] ── Results ──────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<40} {v:.4f}" if isinstance(v, float) else f"  {k:<40} {v}")

    with open(config.EVAL_OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {config.EVAL_OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    evaluate(num_samples=100)
