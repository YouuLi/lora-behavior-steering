---
pretty_name: LoRA Detail and Register Behavior Endpoints
language:
- en
license: other
task_categories:
- text-generation
tags:
- lora
- controllable-generation
- style-steering
- instruction-tuning
configs:
- config_name: detail-target-1
  data_files:
  - split: train
    path: data/alpaca_detail_score1_1k.jsonl
- config_name: detail-target-5
  data_files:
  - split: train
    path: data/alpaca_detail_score5_1k.jsonl
- config_name: register-target-1
  data_files:
  - split: train
    path: data/alpaca_register_score1_1k.jsonl
- config_name: register-target-5
  data_files:
  - split: train
    path: data/alpaca_register_score5_1k.jsonl
---

# LoRA Detail and Register Behavior Endpoints

This dataset contains paired English instruction-tuning responses used to train LoRA
endpoints for two response-level behavior axes:

- **detail**: highly compressed to highly explanatory responses;
- **register**: formal/professional to conversational/natural wording.

The files use Alpaca-format fields:

```json
{"instruction": "...", "input": "", "output": "..."}
```

## Dataset files

| Configuration | File | Intended endpoint |
|---|---|---|
| `detail-target-1` | `alpaca_detail_score1_1k.jsonl` | compressed, low-detail response |
| `detail-target-5` | `alpaca_detail_score5_1k.jsonl` | explanatory, high-detail response |
| `register-target-1` | `alpaca_register_score1_1k.jsonl` | formal/professional register |
| `register-target-5` | `alpaca_register_score5_1k.jsonl` | conversational/natural register |

Files belonging to the same axis retain aligned instructions so that low and high
endpoints can be trained as paired contrasts.

## Important interpretation of the labels

The numbers `1` and `5` are **generation targets**, not guarantees that every response
receives exactly that score from an independent evaluator. In an existing audit of the
first 120 detail examples per side using Qwen2.5-7B-Instruct:

| Intended endpoint | Mean judged detail score | Exact target rate |
|---|---:|---:|
| detail target 1 | 1.59 | 52.5% scored exactly 1 |
| detail target 5 | 4.67 | 72.5% scored exactly 5 |

The two distributions are strongly separated, but they are not perfectly pure classes.
Users studying ordinal detail levels should independently rescore the full dataset or
filter examples under their own rubric. Some long responses may require additional
truncation checks.

## Intended use

- training paired LoRA response-behavior endpoints;
- studying effective LoRA weight-update geometry;
- evaluating continuous detail/register steering;
- testing cross-seed direction extraction and composition.

## Limitations

- The dataset is generated and may contain factual errors, unsupported elaboration, or
  stylistic artifacts.
- Target levels are ordinal prompts rather than human-verified gold labels.
- Detail and register may remain partially coupled.
- Users should check core-content preservation, factuality, relevance, repetition, and
  truncation before using individual examples.
- The data are not intended as a factual knowledge source or for high-stakes decisions.

## Reproduction code

Code and configuration templates are available at:

`https://github.com/YouuLi/lora-behavior-steering`

## License

Before publishing, replace the `license: other` field with a license that is compatible
with the source instructions and the model used to generate the responses. Document the
generator model, source dataset license, and any redistribution requirements here.

## Citation

Please cite the accompanying repository and paper. Publication metadata will be added
after acceptance.

