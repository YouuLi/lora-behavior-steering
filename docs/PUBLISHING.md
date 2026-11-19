# Publishing GitHub code and Hugging Face data

## 1. Finalize repository identifiers

The published identifiers are:

- GitHub: `YouuLi/lora-behavior-steering`
- Hugging Face: `LIYOUYANG/lora-behavior-endpoints`

The GitHub and Hugging Face account names may differ.

## 2. Verify redistribution terms

The current dataset card intentionally uses `license: other`. Before making the dataset
public, verify the license of the source instructions and the terms of the generator
model. Replace the license field and document provenance. A generated response does not
automatically remove restrictions attached to the source prompts.

## 3. Validate the endpoint files

```powershell
python scripts\validate_hf_dataset.py `
  --data-dir "D:\LoRA\lora_weight_space\llama_factory\data"
```

This verifies JSONL syntax, required Alpaca fields, non-empty outputs, duplicate keys,
and positional pairing of the two endpoints on each axis.

## 4. Authenticate with Hugging Face

Use either the CLI login or a session-scoped environment variable:

```powershell
huggingface-cli login
```

or:

```powershell
$env:HF_TOKEN = "your-write-token"
```

Never commit a token to GitHub or place it in a configuration file.

## 5. Upload the dataset

Start with a private repository while checking the viewer and dataset card:

```powershell
python scripts\upload_dataset_to_hf.py `
  --repo-id LIYOUYANG/lora-behavior-endpoints `
  --data-dir "D:\LoRA\lora_weight_space\llama_factory\data" `
  --private
```

After verifying the files, metadata, provenance, and license on Hugging Face, change the
repository visibility in its settings.

## 6. Publish the code to GitHub

```powershell
git init
git add .
git commit -m "Initial release of LoRA behavior steering"
git branch -M main
git remote add origin https://github.com/YouuLi/lora-behavior-steering.git
git push -u origin main
```

The full JSONL files, adapters, dense directions, and generated outputs are ignored by
Git. The GitHub repository keeps only code, documentation, tests, and small examples.

