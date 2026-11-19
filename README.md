# LoRA Behavior Steering in Effective Weight-Update Space

Reference implementation for extracting and composing **detail** and **register**
directions from independently trained LoRA adapters. The method converts each LoRA
adapter into its effective dense update,

\[
\Delta W = \frac{\alpha_{\mathrm{LoRA}}}{r}BA,
\]

then estimates cross-seed behavior directions and injects them into a base model:

\[
W' = W + c_d v_{\mathrm{detail}} + c_r v_{\mathrm{register}}.
\]

No detail or register instruction is appended to the user prompt at inference time.

## What is included

- Effective `Delta-W` export from PEFT LoRA adapters.
- Three reproducible direction estimators:
  - normalized mean paired endpoint difference (recommended);
  - uncentered PCA/SVD of paired endpoint differences;
  - pooled-endpoint centered PCA with label-assisted component selection, matching the
    original experiments.
- Direct single-axis and two-axis base-model steering.
- Lightweight word, sentence, repetition, instability, and token-limit diagnostics.
- LLaMA-Factory configuration and endpoint-manifest examples.
- Unit tests for the direction mathematics and stability diagnostics.

Large adapters, dense direction tensors, training responses, and generated outputs are
excluded from Git by default.

## Data availability

The endpoint training data are distributed separately through Hugging Face Datasets:

[`LIYOUYANG/lora-behavior-endpoints`](https://huggingface.co/datasets/LIYOUYANG/lora-behavior-endpoints)

The GitHub repository contains code, configuration templates, mathematical definitions,
tests, and small example prompts. It does not duplicate the full JSONL training files.
See [`hf_dataset/README.md`](hf_dataset/README.md) for the dataset card and
[`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the upload workflow.

## Repository layout

```text
.
|-- configs/
|   |-- endpoints.example.json
|   |-- dataset_info.example.json
|   `-- llamafactory_train.example.yaml
|-- data/
|   |-- README.md
|   `-- sample_prompts.jsonl
|-- src/lora_behavior_steering/
|   |-- cli_extract.py
|   |-- cli_generate.py
|   |-- directions.py
|   |-- effective_updates.py
|   |-- metrics.py
|   `-- steering.py
|-- tests/
|-- requirements.txt
`-- pyproject.toml
```

## Installation

Python 3.10 or newer and a CUDA-enabled PyTorch installation are recommended.

```powershell
git clone https://github.com/YouuLi/lora-behavior-steering.git
Set-Location lora-behavior-steering
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Install LLaMA-Factory separately when training the endpoint adapters. The extraction
and inference code only requires the packages declared by this project.

## 1. Prepare behavior endpoints

Train two endpoints for each behavior axis using the same instructions and training
settings across seeds.

| Axis | Low endpoint | High endpoint | Controlled content |
|---|---|---|---|
| Detail | detail score 1 | detail score 5 | Preserve the core answer and register |
| Register | formal | conversational | Preserve content and approximate length |

The example configuration in
[`configs/llamafactory_train.example.yaml`](configs/llamafactory_train.example.yaml)
uses Qwen2.5-0.5B-Instruct, LoRA rank 8, and `q_proj,v_proj`. Create one configuration
per endpoint and seed, changing `dataset`, `output_dir`, and `seed`.

To generate all 12 detail/register endpoint configurations for seeds 7, 42, and 123:

```powershell
python scripts\make_train_configs.py
```

After placing the four datasets in LLaMA-Factory and registering them using
`configs/dataset_info.example.json`, the PowerShell helper can run every configuration:

```powershell
.\scripts\train_all.ps1
```

Generated labels are not accepted at face value. Independently audit endpoint responses
because a requested score-1 or score-5 answer may receive a different measured score.

Download the data repository with:

```powershell
hf download LIYOUYANG/lora-behavior-endpoints `
  --repo-type dataset `
  --local-dir data\endpoint_data
```

## 2. Describe the adapters

Copy and edit the example manifest:

```powershell
Copy-Item configs\endpoints.example.json configs\endpoints.json
```

All paths may be absolute or relative to the current working directory. Each low and
high endpoint must have the same seed keys.

## 3. Extract directions

The recommended estimator is the normalized mean paired difference:

\[
v = \operatorname{normalize}\left[
\frac{1}{N}\sum_{i=1}^{N}
(\Delta W_{i,\mathrm{high}}-\Delta W_{i,\mathrm{low}})
\right].
\]

```powershell
lora-steering-extract `
  --manifest configs\endpoints.json `
  --axis detail `
  --method mean_difference `
  --output vectors\detail.safetensors

lora-steering-extract `
  --manifest configs\endpoints.json `
  --axis register `
  --method mean_difference `
  --output vectors\register.safetensors
```

Alternative methods are available through:

```text
--method uncentered_difference_pca
--method endpoint_pca
```

Do not apply standard centered PCA directly to paired differences when the goal is the
shared mean behavior contrast. Centering the differences subtracts that contrast and
instead extracts variation among random seeds.

### Exact pooled-endpoint PCA criterion

The `endpoint_pca` option reproduces the original implementation. All `2N` low and high
endpoint updates are centered by their global mean and decomposed through their Gram
matrix. For component `k`, the projected low and high scores are used to compute

\[
J_k = \frac{\bar z_{\mathrm{high}}^{(k)}-\bar z_{\mathrm{low}}^{(k)}}
{\sqrt{(\operatorname{Var}(z_{\mathrm{low}}^{(k)})+
\operatorname{Var}(z_{\mathrm{high}}^{(k)}))/2}}.
\]

A component is eligible when it explains at least 1% of variance and every paired
high-minus-low projection has the same sign across seeds. The eligible component with
maximum `|J_k|` is selected and oriented toward the high endpoint. If none passes the
sign-consistency test, the same selection is applied to all components above the
variance threshold. This is a label-assisted component selection procedure, rather
than fully unsupervised PCA.

## 4. Generate with two-axis steering

```powershell
lora-steering-generate `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --detail-vector vectors\detail.safetensors `
  --register-vector vectors\register.safetensors `
  --prompts data\sample_prompts.jsonl `
  --alpha 1.0 `
  --gamma 1.0 `
  --max-new-tokens 512 `
  --output outputs\alpha_1_gamma_1.jsonl
```

`alpha` controls the detail direction and `gamma` controls the register direction.
Useful coefficient ranges are model- and direction-specific. Sweep coefficients on a
held-out prompt set and report failure rates; do not assume that extrapolation beyond
the trained endpoints is stable.

A complete grid can be generated in one run:

```powershell
lora-steering-generate `
  --detail-vector vectors\detail.safetensors `
  --register-vector vectors\register.safetensors `
  --prompts data\sample_prompts.jsonl `
  --alphas=-1,-0.5,0,0.5,1 `
  --gammas=-1,-0.5,0,0.5,1 `
  --output outputs\two_axis_grid.jsonl
```

## Evaluation

The command-line generator records basic diagnostics. A paper-level evaluation should
add axis-aware measures:

- detail score and blind pairwise detail preference;
- register score, blind pairwise register preference, and `P(formal)`;
- core-content preservation and candidate-to-base entailment;
- factuality, relevance, and information gain for detail expansion;
- bidirectional contradiction, repetition, instability, and truncation.

Detail steering can add valid information, so symmetric embedding similarity alone is
not a sufficient fidelity measure. Register steering is closer to a meaning-preserving
transformation and should retain propositional content in both directions.

## Reproducibility notes

- Pair endpoints by seed and keep all LoRA hyperparameters fixed.
- Analyze the effective update `B @ A`, not the individual factors; equivalent factor
  pairs can represent the same dense update.
- Record endpoint paths, seed IDs, estimator, coefficient grid, decoding parameters,
  and judge versions.
- Use a held-out prompt set for steering evaluation.
- With only three seeds, geometric conclusions should be presented as empirical
  evidence rather than universal properties of LoRA space.

## Tests

```powershell
pytest -q
```

## Citation

If this repository supports your work, cite the accompanying paper. Replace the
placeholder fields in [`CITATION.cff`](CITATION.cff) when the final publication record
is available.

## License

Released under the MIT License. Training data, base models, and third-party checkpoints
remain subject to their respective licenses.

