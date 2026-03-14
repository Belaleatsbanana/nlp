import argparse
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from rouge_score import rouge_scorer

import config
from load_dataset import load_raw_df, df_to_dataset_dict


# BERTScore overflow patch
import transformers as _transformers

_original_from_pretrained = _transformers.AutoTokenizer.from_pretrained.__func__

@classmethod
def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    tokenizer = _original_from_pretrained(
        cls, pretrained_model_name_or_path, *args, **kwargs
    )
    _BERTSCORE_MAX = 512
    if getattr(tokenizer, "model_max_length", 0) > _BERTSCORE_MAX:
        tokenizer.model_max_length = _BERTSCORE_MAX
    return tokenizer

_transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

from bert_score import score as bertscore


# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen2.5 medical model")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to the adapter directory (e.g., ./qwen-medical-adapter-dora)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path where evaluation results (JSON) will be saved")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate (default: 100)")
    return parser.parse_args()


# Model loading

def load_eval_model(adapter_dir):
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
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# Generation

def generate_answer(model, tokenizer, question: str) -> str:
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
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            num_beams=config.NUM_BEAMS,
            do_sample=(config.NUM_BEAMS == 1),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Medical accuracy heuristic

def medical_accuracy_score(prediction: str, reference: str) -> float:
    ref_lower  = reference.lower()
    pred_lower = prediction.lower()

    relevant_keywords = [
        kw for kw in config.CLINICAL_GUIDELINE_KEYWORDS if kw in ref_lower
    ]
    if not relevant_keywords:
        return 1.0

    matched = sum(1 for kw in relevant_keywords if kw in pred_lower)
    return matched / len(relevant_keywords)


# ROUGE-L

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return {"rougeL_mean": sum(scores) / len(scores) if scores else 0.0}


# Main evaluation loop

def evaluate(adapter_dir, output_file, num_samples: int):
    print(f"[eval] Loading model from {adapter_dir}…")
    model, tokenizer = load_eval_model(adapter_dir)

    print("[eval] Loading test set from CSV…")
    df  = load_raw_df()
    dsd = df_to_dataset_dict(df)
    raw = dsd[config.DATASET_TEST_SPLIT]
    if num_samples:
        raw = raw.select(range(min(num_samples, len(raw))))

    questions  = list(raw[config.DATASET_QUESTION_COL])
    references = list(raw[config.DATASET_ANSWER_COL])

    predictions    = []
    med_acc_scores = []

    print(f"[eval] Generating answers for {len(questions)} samples…")
    for q, ref in tqdm(zip(questions, references), total=len(questions)):
        pred = generate_answer(model, tokenizer, q)
        predictions.append(pred)
        med_acc_scores.append(medical_accuracy_score(pred, ref))

    # BERTScore
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

    # ROUGE-L
    print("[eval] Computing ROUGE-L…")
    rouge_results = compute_rouge(predictions, references)

    # Medical accuracy
    med_acc_mean = sum(med_acc_scores) / len(med_acc_scores)

    results = {
        **bert_results,
        **rouge_results,
        "medical_accuracy_keyword_mean": med_acc_mean,
        "num_samples": len(questions),
    }

    print("\n[eval] Results")
    for k, v in results.items():
        print(f"  {k:<40} {v:.4f}" if isinstance(v, float) else f"  {k:<40} {v}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {output_file}")

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.adapter_dir, args.output_file, args.num_samples)