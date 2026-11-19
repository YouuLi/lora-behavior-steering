from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import bigram_repetition, content_unstable, sentence_count, word_count
from .steering import temporary_steering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with detail and register weight steering.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--detail-vector", required=True)
    parser.add_argument("--register-vector", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL with prompt or instruction field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.0, help="Detail coefficient")
    parser.add_argument("--gamma", type=float, default=0.0, help="Register coefficient")
    parser.add_argument("--alphas", help="Optional comma-separated detail grid")
    parser.add_argument("--gammas", help="Optional comma-separated register grid")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
    ).eval()
    detail = load_file(args.detail_vector, device="cpu")
    register = load_file(args.register_vector, device="cpu")
    prompts = [json.loads(line) for line in Path(args.prompts).open(encoding="utf-8") if line.strip()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alphas = [float(value) for value in args.alphas.split(",")] if args.alphas else [args.alpha]
    gammas = [float(value) for value in args.gammas.split(",")] if args.gammas else [args.gamma]
    total = len(prompts) * len(alphas) * len(gammas)
    completed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for alpha in alphas:
            for gamma in gammas:
                with temporary_steering(model, [(alpha, detail), (gamma, register)]):
                    for index, row in enumerate(prompts):
                        prompt = row.get("prompt") or row.get("instruction")
                        if not prompt:
                            raise ValueError(f"Row {index} has no prompt or instruction")
                        messages = [{"role": "user", "content": prompt}]
                        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer([chat], return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            generated = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
                        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        record = {
                            **row,
                            "prompt": prompt,
                            "alpha": alpha,
                            "gamma": gamma,
                            "response": response,
                            "word_count": word_count(response),
                            "sentence_count": sentence_count(response),
                            "bigram_repetition": bigram_repetition(response),
                            "content_unstable": int(content_unstable(response)),
                            "hit_token_limit": int(len(new_tokens) >= args.max_new_tokens),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed += 1
                        print(f"Generated {completed}/{total} (alpha={alpha}, gamma={gamma})")


if __name__ == "__main__":
    main()

