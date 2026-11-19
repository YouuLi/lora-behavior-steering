from __future__ import annotations

from pathlib import Path

SEEDS = (7, 42, 123)
ENDPOINTS = {
    "detail_score1": "alpaca_detail_score1_1k",
    "detail_score5": "alpaca_detail_score5_1k",
    "register_score1": "alpaca_register_score1_1k",
    "register_score5": "alpaca_register_score5_1k",
}

TEMPLATE = """model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

dataset: {dataset}
template: qwen
cutoff_len: 768
max_samples: 844
overwrite_cache: true
preprocessing_num_workers: 2

output_dir: saves/qwen25_05b/lora/{endpoint}_seed{seed}
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
fp16: true
seed: {seed}
val_size: 0
"""


def main() -> None:
    output = Path("configs/generated")
    output.mkdir(parents=True, exist_ok=True)
    for endpoint, dataset in ENDPOINTS.items():
        for seed in SEEDS:
            path = output / f"train_{endpoint}_seed{seed}.yaml"
            path.write_text(
                TEMPLATE.format(endpoint=endpoint, dataset=dataset, seed=seed),
                encoding="utf-8",
            )
            print(path)


if __name__ == "__main__":
    main()

