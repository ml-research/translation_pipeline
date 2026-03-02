"""CLI entrypoint for the Datatrove translation pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .pipeline_ops import *  # shared pipeline/runtime functions and constants


def main() -> int:
    parser = argparse.ArgumentParser(description="Datatrove translation pipeline for LongBench and RULER")
    parser.add_argument("--benchmark", choices=["longbench", "ruler"], required=True)
    parser.add_argument("--language", type=str, required=True, help="Target language code (e.g., de, fr, pl, en)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for vLLM endpoint")
    parser.add_argument("--input-dir", type=str, default="./datatrove_inputs")
    parser.add_argument("--output-dir", type=str, default="./datatrove_outputs")
    parser.add_argument("--checkpoints-dir", type=str, default="./datatrove_checkpoints")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    parser.add_argument("--presence-penalty", type=float, default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="Enable model thinking mode. Translation jobs should typically keep this disabled.",
    )
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--max-concurrent-generations", type=int, default=16)
    parser.add_argument("--max-concurrent-documents", type=int, default=None)
    parser.add_argument("--records-per-chunk", type=int, default=200)
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=120, help="Seconds between progress logs (0 disables)")
    parser.add_argument("--clean-output", action="store_true", help="Delete output/checkpoint/logs before run")
    parser.add_argument(
        "--retry-failed-from",
        type=str,
        default=None,
        help="Only process rows that previously had _translation_failed in this output dir",
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--save-to-disk", action="store_true", help="Save HF dataset to disk after translation")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN)")
    parser.add_argument("--hf-org", type=str, default="AIML-TUDA")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--no-skip-completed", action="store_true", help="Disable datatrove skip-completed behavior")
    parser.add_argument(
        "--save-failed-attempts",
        "--persist-failed-artifacts",
        dest="save_failed_attempts",
        action="store_true",
        default=True,
        help=(
            "Persist failed translation attempt details (including rejected candidate text) to logging dir. "
            "--persist-failed-artifacts is kept as a legacy alias."
        ),
    )
    parser.add_argument(
        "--no-save-failed-attempts",
        "--no-persist-failed-artifacts",
        dest="save_failed_attempts",
        action="store_false",
        help="Disable persistence of failed translation attempt details (legacy alias supported).",
    )
    parser.add_argument(
        "--failed-attempts-filename",
        type=str,
        default=FAILED_ATTEMPTS_FILENAME,
        help="Filename for failed attempt JSONL within each config logging dir.",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Comma-separated list or substring filter for LongBench subjects",
    )
    parser.add_argument(
        "--ruler-dataset",
        type=str,
        default=DEFAULT_RULER_DATASET,
        help="RULER dataset name",
    )
    parser.add_argument(
        "--ruler-splits",
        type=str,
        default=",".join(DEFAULT_RULER_SPLITS),
        help="Comma-separated list of RULER splits",
    )

    args = parser.parse_args()
    language = args.language.strip()
    benchmark = args.benchmark

    if args.presence_penalty < 0 or args.presence_penalty > 2:
        parser.error("--presence-penalty must be between 0 and 2.")

    if args.enable_thinking:
        logger.warning(
            "Thinking mode enabled. Qwen recommendation: --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0."
        )

    if language not in DEFAULT_LANGUAGES:
        logger.warning("Language %s is not in the default EU5+PL set", language)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    hf_repo = args.hf_repo
    if hf_repo is None:
        if benchmark == "longbench":
            hf_repo = f"{args.hf_org}/LongBench-multilingual"
        else:
            hf_repo = f"{args.hf_org}/RULER-multilingual"

    if language != "en" and args.model_name is None:
        logger.info("Auto-detecting model name from %s...", args.vllm_url)
        model_name = detect_model_name(args.vllm_url)
        logger.info("Detected model name: %s", model_name)
    else:
        model_name = args.model_name

    skip_completed = not args.no_skip_completed

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir) / benchmark
    checkpoints_base = Path(args.checkpoints_dir) / benchmark
    clean_output = args.clean_output

    retry_root = Path(args.retry_failed_from) if args.retry_failed_from else None
    retry_single_config = None
    if retry_root and list(retry_root.glob("*.jsonl*")):
        retry_single_config = retry_root.name

    if benchmark == "longbench":
        output_mode = OUTPUT_MODE_REPLACE
        language_key = "language"
        include_source_language = True
        include_lang_fields = False
        include_failed = True
        subjects = LONGBENCH_DATASETS
        if args.dataset_filter:
            filter_items = [item.strip().lower() for item in args.dataset_filter.split(",")]
            subjects = [s for s in subjects if subject_matches_filter(s, filter_items)]
            logger.info("Filtered to %d LongBench subjects: %s", len(subjects), subjects)

        cache_dir = input_dir / "longbench" / ".cache"
        data_dir = download_and_extract_longbench(cache_dir)

        for subject in subjects:
            output_subject = canonical_subject_name(subject)
            config_name = f"{output_subject}_{language}"
            logger.info("Processing LongBench/%s -> %s", subject, config_name)

            input_path = data_dir / f"{subject}.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing LongBench data file: {input_path}")

            if args.max_examples:
                sample_dir = input_dir / "longbench" / "samples"
                sample_name = f"{subject}_sample_{args.max_examples}.jsonl"
                input_path = sample_jsonl(input_path, sample_dir / sample_name, args.max_examples)

            output_dir = output_base / config_name
            logging_dir = checkpoints_base / config_name / "logs"
            checkpoints_dir = checkpoints_base / config_name / "checkpoints"

            retry_ids = None
            if retry_root:
                if retry_single_config:
                    if config_name != retry_single_config:
                        logger.info("Skipping %s (retry scope is %s)", config_name, retry_single_config)
                        continue
                    retry_dir = retry_root
                else:
                    retry_dir = retry_root / config_name
                    if not retry_dir.exists():
                        logger.info("Skipping %s (no retry output at %s)", config_name, retry_dir)
                        continue
                retry_ids = collect_failed_row_ids(retry_dir)
                logger.info("Retrying %d failed rows from %s", len(retry_ids), retry_dir)
                if not retry_ids:
                    logger.info("No failed rows found for %s; skipping.", config_name)
                    continue

            run_datatrove_translation(
                input_path=input_path,
                output_dir=output_dir,
                logging_dir=logging_dir,
                checkpoints_dir=checkpoints_dir,
                language=language,
                fields_to_translate=LONGBENCH_FIELDS,
                vllm_url=args.vllm_url,
                model_name=model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                enable_thinking=args.enable_thinking,
                request_timeout=args.request_timeout,
                max_concurrent_generations=args.max_concurrent_generations,
                max_concurrent_documents=args.max_concurrent_documents,
                records_per_chunk=args.records_per_chunk,
                retry_count=args.retry_count,
                output_mode=output_mode,
                language_key=language_key,
                include_source_language=include_source_language,
                include_lang_fields=include_lang_fields,
                include_failed=include_failed,
                skip_completed=skip_completed,
                clean_output=clean_output,
                progress_interval=args.progress_interval,
                progress_label=config_name,
                retry_ids=retry_ids,
                translation_profile=TRANSLATION_PROFILE_DEFAULT,
                save_failed_attempts=args.save_failed_attempts,
                failed_attempts_filename=args.failed_attempts_filename,
            )

            if not args.no_upload and hf_token:
                upload_to_hub(
                    output_dir=output_dir,
                    hf_repo=hf_repo,
                    config_name=config_name,
                    hf_token=hf_token,
                    save_to_disk=args.save_to_disk,
                    split_name=args.split_name,
                )

    elif benchmark == "ruler":
        output_mode = OUTPUT_MODE_REPLACE
        language_key = "language"
        include_source_language = True
        include_lang_fields = False
        include_failed = True
        splits = parse_comma_list(args.ruler_splits)
        cache_dir = input_dir / "ruler"

        for split in splits:
            config_name = f"{split}_{language}"
            logger.info("Processing RULER/%s -> %s", split, config_name)

            input_path = ensure_ruler_jsonl(args.ruler_dataset, split, cache_dir, args.max_examples)
            output_dir = output_base / config_name
            logging_dir = checkpoints_base / config_name / "logs"
            checkpoints_dir = checkpoints_base / config_name / "checkpoints"

            retry_ids = None
            if retry_root:
                if retry_single_config:
                    if config_name != retry_single_config:
                        logger.info("Skipping %s (retry scope is %s)", config_name, retry_single_config)
                        continue
                    retry_dir = retry_root
                else:
                    retry_dir = retry_root / config_name
                    if not retry_dir.exists():
                        logger.info("Skipping %s (no retry output at %s)", config_name, retry_dir)
                        continue
                retry_ids = collect_failed_row_ids(retry_dir)
                logger.info("Retrying %d failed rows from %s", len(retry_ids), retry_dir)
                if not retry_ids:
                    logger.info("No failed rows found for %s; skipping.", config_name)
                    continue

            run_datatrove_translation(
                input_path=input_path,
                output_dir=output_dir,
                logging_dir=logging_dir,
                checkpoints_dir=checkpoints_dir,
                language=language,
                fields_to_translate=RULER_FIELDS,
                vllm_url=args.vllm_url,
                model_name=model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                enable_thinking=args.enable_thinking,
                request_timeout=args.request_timeout,
                max_concurrent_generations=args.max_concurrent_generations,
                max_concurrent_documents=args.max_concurrent_documents,
                records_per_chunk=args.records_per_chunk,
                retry_count=args.retry_count,
                output_mode=output_mode,
                language_key=language_key,
                include_source_language=include_source_language,
                include_lang_fields=include_lang_fields,
                include_failed=include_failed,
                skip_completed=skip_completed,
                clean_output=clean_output,
                progress_interval=args.progress_interval,
                progress_label=config_name,
                retry_ids=retry_ids,
                translation_profile=(
                    TRANSLATION_PROFILE_RULER_NIAH
                    if split.startswith("niah")
                    else TRANSLATION_PROFILE_RULER_QA
                    if split.startswith("qa_")
                    else TRANSLATION_PROFILE_DEFAULT
                ),
                save_failed_attempts=args.save_failed_attempts,
                failed_attempts_filename=args.failed_attempts_filename,
            )

            if not args.no_upload and hf_token:
                upload_to_hub(
                    output_dir=output_dir,
                    hf_repo=hf_repo,
                    config_name=config_name,
                    hf_token=hf_token,
                    save_to_disk=args.save_to_disk,
                    split_name=args.split_name,
                )

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    return 0
