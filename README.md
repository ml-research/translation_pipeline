# translation_pipeline

Datatrove-based translation pipeline for benchmark datasets (LongBench and RULER), with vLLM inference, retry handling, and optional Hugging Face upload.
It is also designed for other long-context datasets where chunk-first translation helps prevent models from answering tasks instead of translating long reasoning traces or long-form text.

## Requirements

- Python `>=3.12`
- `uv`
- Reachable vLLM OpenAI-compatible endpoint (default: `http://localhost:8000/v1`)

Optional workspace setting:

```bash
export UV_PYTHON_INSTALL_DIR=/mnt/vast/workspaces/jackal_ai/.uv/python
```

Install dependencies:

```bash
uv sync
```

## How It Works

1. Input loading:
   - LongBench: downloads/extracts data files and processes subject-wise JSONL.
   - RULER: loads configured split(s) from the HF dataset and writes JSONL inputs.
2. Datatrove pipeline setup:
   - Reads rows as `Document`s, preserving row metadata.
   - Processes in chunks (`records-per-chunk`) with configurable concurrency.
   - Chunking is intentional for long contexts: many translation models tend to switch into answer-generation mode instead of faithful translation when prompts become very long.
3. Field translation:
   - LongBench fields: `input`, `context`, `answers`
   - RULER fields: `input`, `outputs`
   - Uses specialized profiles for RULER QA/NIAH layouts to reduce malformed generations.
   - The same chunk-first strategy is useful beyond LongBench/RULER for any dataset with long reasoning traces or other long-form text that must be translated without changing task semantics.
4. Failure handling:
   - Retries translation attempts per row.
   - Marks unrecoverable rows with `_translation_failed=True`.
   - Persists failed attempt artifacts to per-config logs.
5. Output and publishing:
   - Writes JSONL outputs per benchmark config.
   - Optionally uploads to Hugging Face and/or saves dataset artifacts to disk.

## Usage

Main entrypoint:

```bash
python translate_datatrove.py --help
```

### LongBench example

```bash
python translate_datatrove.py \
  --benchmark longbench \
  --language de \
  --vllm-url http://localhost:8000/v1 \
  --output-dir ./datatrove_outputs \
  --checkpoints-dir ./datatrove_checkpoints \
  --dataset-filter hotpotqa,musique
```

### RULER example

```bash
python translate_datatrove.py \
  --benchmark ruler \
  --language de \
  --vllm-url http://localhost:8000/v1 \
  --ruler-splits qa_1,qa_2 \
  --output-dir ./datatrove_outputs \
  --checkpoints-dir ./datatrove_checkpoints
```

### Retry previously failed rows

```bash
python translate_datatrove.py \
  --benchmark longbench \
  --language de \
  --retry-failed-from ./datatrove_outputs/longbench/hotpotqa_de \
  --output-dir ./datatrove_outputs \
  --checkpoints-dir ./datatrove_checkpoints
```

### Hugging Face upload

Set `HF_TOKEN` or pass `--hf-token`. Upload is enabled by default unless `--no-upload` is set.

## Validation

Use `validate_datatrove_output.py` to check missing/empty fields, failure flags, and length ratios against English:

```bash
python validate_datatrove_output.py \
  --output-dir ./datatrove_outputs/longbench/hotpotqa_de \
  --language de \
  --fields input,context,answers \
  --no-suffix \
  --compare-dir ./datatrove_outputs/longbench/hotpotqa_en
```

## Repository Layout

- `translate_datatrove.py`: compatibility wrapper and CLI entrypoint.
- `translation_pipeline/cli.py`: argument parsing and benchmark orchestration.
- `translation_pipeline/general_logic.py`: shared translation and validation logic.
- `translation_pipeline/ruler_logic.py`: RULER-specific parsing/profiles.
- `translation_pipeline/longbench_logic.py`: LongBench-specific dataset helpers.
- `translation_pipeline/pipeline_ops.py`: Datatrove runtime, IO, retry merge, and upload utilities.
- `validate_datatrove_output.py`: post-run output validator.
