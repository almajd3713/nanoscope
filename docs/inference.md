# Generate text from a checkpoint

Inference continues a prompt from a saved Nanoscope checkpoint. It does not start
training, load training data, or initialize HF/W&B logging. The model must be
registered in the installed project, just as it must be for evaluation.

## Try the local smoke example

```bash
uv run nanoscope train --config configs/test/eval/local-base.yaml --resume auto
uv run nanoscope generate --config configs/test/inference/local.yaml --prompt "Hello"
```

The command prints the prompt and continuation. It also appends a JSON object with
`prompt`, `completion`, `text` (both combined), token IDs, and provenance to
`runs/generations/eval-local-base.jsonl`. Add `--json` to print the full records as
JSONL instead of readable text. The smoke model is `toy_lm`; its output
tests the inference flow and is not expected to be coherent language.

## Use your own model

Copy [the commented template](../configs/inference-template.yaml) to
`configs/inference/my-run.yaml`. Set `run_dir` to the run containing `checkpoints/`.
`checkpoint: latest` selects the highest complete step, skipping corrupt or partial
checkpoints. Use an explicit checkpoint directory to pin a comparison to a step:

```yaml
checkpoint: "@/runs/my-run/checkpoints/step_00001000"
```

Explicit checkpoint paths are YAML-relative unless absolute or prefixed by `@/`;
they are not relative to `run_dir`. An explicit path overrides `run_dir`.
The loader verifies checksums, run/step metadata, config digest, and tokenizer
identity when recorded. Older checkpoints can use `training_config` to supply
their original resolved config. Use checkpoints produced by your own training
workflow; these are PyTorch training snapshots, not model-only exports.

```bash
uv run nanoscope generate --config configs/inference/my-run.yaml \
  --prompt "The reason the sky appears blue is"
```

The checkpoint defines the architecture, tokenizer, and context length. Inference
uses FP32 on one CPU or GPU, with dropout disabled and gradients off. `device: auto`
selects CUDA if available; `cpu` and `cuda` request a specific type. The loader reads
the snapshot on CPU, releases unused training state, then moves only the model to
the selected device.

GPT-2 tokenization needs its cached vocabulary assets; the first use may download
them through tiktoken. The byte-tokenizer smoke example runs offline.

## Repeat prompts and compare increments

```bash
uv run nanoscope generate --config configs/inference/my-run.yaml \
  --prompts configs/test/inference/prompts.jsonl
```

Prompt files contain one **JSON string** per line, preserving multiline prompts
using `\n` and empty prompts using `""`. All prompts are checked before generation;
the model is loaded once. CLI prompt-file paths are relative to the current directory.

Use the same prompt file, sampling settings, and training budget for each variant.
Give each variant its own output path. Each sample records the checkpoint hash/step,
config digest, tokenizer identity, model/generator source hashes, sampling settings,
context handling, and runtime. The seed resets per prompt, so reordering a prompt
file does not change a prompt's random stream. Reproducibility assumes the same
model code, software, device and numerical settings; a seed does not promise
identical output across different hardware.

Outputs append on each run. `output: null` disables saving. These samples complement
the frozen-corpus evaluator; they do not replace validation loss or enter its reports.

## Load once in Python or a notebook

```python
from nanoscope.inference import GenerationSession

session = GenerationSession.from_config("configs/inference/my-run.yaml")
print(session.generate("Once upon a time")["text"])
print(session.generate("A different prompt", temperature=0, max_new_tokens=64)["text"])
```

Each call can override `temperature`, `top_p`, `max_new_tokens`, and `seed`.
Overrides are validated and saved with that sample without changing defaults.

In the [Kaggle runner](../notebooks/kaggle-runner.ipynb), enable `RUN_INFERENCE` and
select `INFERENCE_CONFIG`. After training, run **Load a model for inference** once
and rerun **Try prompts** as often as you like. The default inference config matches
the default Kaggle smoke run. Samples are saved separately in
`/kaggle/working/nanoscope-exports/` for download through the output file browser.

For inference without training, run workspace preparation and dependency installation,
then the inference cells. Point the config at an attached checkpoint under
`/kaggle/input`, or an existing run under `/kaggle/working`. No secrets are needed to
read local attached checkpoints. The evaluation results ZIP contains no weights;
attach the checkpoint itself, including `state.pt`, `metadata.json`, `manifest.json`,
and any sidecars listed in its manifest. Reload the model after switching checkpoints;
restart the notebook session after replacing already-imported code.

## Generation behavior

- `temperature: 0` chooses the most likely next token; positive values sample.
- `top_p` keeps the smallest highest-probability set reaching that probability mass.
- Generation stops on EOS or after `max_new_tokens`, with a recorded finish reason.
- Long inputs use the most recent context window. Initial prompt truncation is recorded;
  generation slides the window as tokens are added. This version recomputes that window
  each step; it has no KV cache or multi-prompt batching yet.
- An empty prompt starts with EOS as context. EOS is omitted from displayed text but
  retained in generated token IDs. Invalid/incomplete UTF-8 byte sequences display a
  replacement character; saved token IDs preserve the actual output.

This is text completion. A pretrained model does not acquire chat or instruction-following
behavior merely by using a chat-shaped prompt.
