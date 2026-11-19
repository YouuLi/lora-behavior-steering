# Data

Training responses are not bundled in this repository. Prepare paired Alpaca-format
JSONL files with the same instructions at the two endpoints of each axis:

```json
{"instruction": "...", "input": "", "output": "..."}
```

For the detail axis, score 1 and score 5 should differ in explanatory coverage while
preserving the core answer. For the register axis, formal and conversational answers
should preserve content and approximately preserve length. Audit generated endpoint
data with an independent judge; requested rubric labels are not guaranteed to equal
the scores actually achieved by the responses.

Do not commit large datasets or generated outputs unless their licenses and privacy
conditions permit redistribution.

The public dataset is hosted separately on Hugging Face:

[`LIYOUYANG/lora-behavior-endpoints`](https://huggingface.co/datasets/LIYOUYANG/lora-behavior-endpoints)

