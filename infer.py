"""
infer.py — Run inference with the fine-tuned Qwen2.5 medical model.

Supports:
  - Single question (interactive CLI)
  - Batch question list
  - Streaming output via TextStreamer
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel

import config


def load_model(use_adapter: bool = True):
    """
    Load the quantised Qwen2.5 model.

    Args:
        use_adapter: If True, load fine-tuned LoRA adapter (post-training).
                     If False, load the base model only (pre-fine-tune baseline).
    """
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

    if use_adapter:
        model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
        tokenizer = AutoTokenizer.from_pretrained(config.ADAPTER_DIR)
        print(f"[infer] Loaded fine-tuned adapter from {config.ADAPTER_DIR}")
    else:
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        print(f"[infer] Loaded base model from {model_ref}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def infer_stream(model, tokenizer, question: str) -> None:
    """Stream the model response to stdout."""
    text = build_prompt(question, tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print(f"\n[Question] {question}\n[Answer]")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            # Precision Tweak: temperature=0.3 for stricter factual adherence
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            num_beams=config.NUM_BEAMS,
            do_sample=True,
            streamer=streamer,
        )
    print()


def infer_batch(model, tokenizer, questions: list[str]) -> list[str]:
    """Generate answers for a list of questions without streaming."""
    answers = []
    for q in questions:
        text = build_prompt(q, tokenizer)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                num_beams=config.NUM_BEAMS,
                do_sample=True,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(answer)
        print(f"[Q] {q}\n[A] {answer}\n{'─'*60}")

    return answers


def interactive_cli(model, tokenizer):
    """Simple REPL for manual testing."""
    print("\nQwen2.5 Medical Assistant (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        infer_stream(model, tokenizer, question)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5 Medical Inference")
    parser.add_argument("--base",        action="store_true",
                        help="Use base model instead of fine-tuned adapter")
    parser.add_argument("--question",    type=str, default=None,
                        help="Single question (non-interactive)")
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Start interactive CLI (default)")
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapter=not args.base)

    if args.question:
        infer_stream(model, tokenizer, args.question)
    else:
        interactive_cli(model, tokenizer)
