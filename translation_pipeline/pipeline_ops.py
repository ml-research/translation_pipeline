"""Pipeline orchestration, IO helpers, and HF upload utilities."""

from __future__ import annotations

from .logic import *  # re-export shared constants, types, loggers, and translation helpers


def build_passthrough_row(
    row: dict[str, Any],
    target_language: str,
    fields_to_translate: list[str],
    output_mode: str,
    language_key: str,
    include_source_language: bool,
    include_lang_fields: bool,
    include_failed: bool,
) -> dict[str, Any]:
    output = dict(row)
    for field in fields_to_translate:
        if output_mode == OUTPUT_MODE_SUFFIX and include_lang_fields:
            output[f"{field}_{target_language}"] = row.get(field)
        elif output_mode == OUTPUT_MODE_REPLACE:
            output[field] = row.get(field)

    if output_mode == OUTPUT_MODE_REPLACE:
        output.pop("source_language", None)
        output.pop("target_language", None)

    if include_source_language:
        output["source_language"] = "en"

    output[language_key] = target_language
    if include_failed:
        output["_translation_failed"] = False
        output.pop("_translation_failure_reasons", None)
    return output


class PassthroughTranslator(PipelineStep):
    name = "Passthrough translator"
    type = "Transform"

    def __init__(
        self,
        target_language: str,
        fields_to_translate: list[str],
        output_mode: str,
        language_key: str,
        include_source_language: bool,
        include_lang_fields: bool,
        include_failed: bool,
    ):
        super().__init__()
        self.target_language = target_language
        self.fields_to_translate = fields_to_translate
        self.output_mode = output_mode
        self.language_key = language_key
        self.include_source_language = include_source_language
        self.include_lang_fields = include_lang_fields
        self.include_failed = include_failed

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            row = doc.metadata.get("row", {})
            doc.metadata["translation_output"] = build_passthrough_row(
                row,
                self.target_language,
                self.fields_to_translate,
                self.output_mode,
                self.language_key,
                self.include_source_language,
                self.include_lang_fields,
                self.include_failed,
            )
            yield doc


def output_adapter(self, document: Document) -> dict[str, Any]:
    output = document.metadata.get("translation_output")
    if isinstance(output, list):
        output = output[0] if output else None
    if output is None:
        return document.metadata.get("row", {})
    return output


class ProgressJsonlWriter(JsonlWriter):
    def __init__(
        self,
        *args,
        total_docs: int | None = None,
        progress_interval: int = 0,
        progress_label: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.total_docs = total_docs
        self.progress_interval = progress_interval
        self.progress_label = progress_label or ""
        self._count = 0
        self._last_log = time.time()

    def write(self, document: Document, rank: int = 0, **kwargs):
        super().write(document, rank, **kwargs)
        self._count += 1
        if self.progress_interval <= 0:
            return
        now = time.time()
        if now - self._last_log < self.progress_interval:
            return
        self._last_log = now
        label = f"{self.progress_label} " if self.progress_label else ""
        if self.total_docs:
            pct = (self._count / self.total_docs) * 100
            progress_logger.info("Progress %s%d/%d (%.1f%%)", label, self._count, self.total_docs, pct)
        else:
            progress_logger.info("Progress %s%d", label, self._count)


def pick_text_field(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
    return ""


def make_reader_adapter(fields_for_text: list[str]):
    def _adapter(self, data: dict, path: str, id_in_file: int | str):
        row = dict(data)
        doc_id = get_row_id(row) or f"{path}/{id_in_file}"
        text = pick_text_field(row, fields_for_text) or " "
        return {"text": text, "id": doc_id, "metadata": {"row": row}}

    return _adapter


def normalize_vllm_endpoint(vllm_url: str) -> str:
    url = vllm_url.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    return url


def ensure_v1_url(vllm_url: str) -> str:
    url = vllm_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def detect_model_name(vllm_url: str) -> str:
    url = ensure_v1_url(vllm_url) + "/models"
    with urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to fetch models from {url}: {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data", [])
    if not models:
        raise RuntimeError(f"No models found at {url}")
    return models[0].get("id")


def count_jsonl_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    total = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def collect_failed_row_ids(output_dir: Path) -> set[str]:
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No JSONL outputs found in {output_dir}")
    failed_ids: set[str] = set()
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("_translation_failed"):
                    continue
                row_id = get_row_id(row)
                if row_id is None:
                    raise RuntimeError(f"Missing row id in failed row from {path}")
                failed_ids.add(row_id)
    return failed_ids


def collect_existing_row_ids(output_dir: Path) -> set[str]:
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        return set()
    row_ids: set[str] = set()
    missing_ids = 0
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = get_row_id(row)
                if row_id is None:
                    missing_ids += 1
                    continue
                row_ids.add(row_id)
    if missing_ids:
        logger.warning(
            "Found %d output rows without an id; resume skip may be incomplete",
            missing_ids,
        )
    return row_ids


def load_jsonl_rows_by_id(output_dir: Path) -> dict[str, dict[str, Any]]:
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No JSONL outputs found in {output_dir}")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = get_row_id(row)
                if row_id is None:
                    raise RuntimeError(f"Missing row id in retry output row from {path}")
                row_id = str(row_id)
                if row_id in rows_by_id:
                    raise RuntimeError(f"Duplicate row id {row_id} found in retry outputs at {output_dir}")
                rows_by_id[row_id] = row
    return rows_by_id


def merge_retry_outputs_into_existing(
    output_dir: Path,
    retry_output_dir: Path,
    expected_retry_ids: set[str],
) -> None:
    if not retry_output_dir.exists():
        raise FileNotFoundError(f"Retry output directory does not exist: {retry_output_dir}")

    source_files = sorted(output_dir.glob("*.jsonl*"))
    if not source_files:
        raise FileNotFoundError(f"No base output files found to merge into: {output_dir}")

    retry_rows_by_id = load_jsonl_rows_by_id(retry_output_dir)
    expected_ids = {str(row_id) for row_id in expected_retry_ids}
    retry_ids = set(retry_rows_by_id.keys())

    missing_retry_rows = expected_ids - retry_ids
    if missing_retry_rows:
        preview = sorted(missing_retry_rows)[:5]
        raise RuntimeError(
            f"Retry output missing {len(missing_retry_rows)} expected row(s) (examples: {preview})"
        )

    unexpected_retry_rows = retry_ids - expected_ids
    if unexpected_retry_rows:
        preview = sorted(unexpected_retry_rows)[:5]
        progress_logger.warning(
            "Retry output contains %d unexpected row(s); they will be ignored during merge (examples: %s)",
            len(unexpected_retry_rows),
            preview,
        )

    replaced_ids: set[str] = set()
    total_rows = 0
    replaced_rows = 0

    # Place the temporary merge files on the same filesystem when possible to
    # keep the final replacement atomic and avoid EXDEV on /tmp -> NFS moves.
    with tempfile.TemporaryDirectory(prefix="datatrove_retry_merge_", dir=output_dir) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for src_path in source_files:
            dst_path = tmp_root / src_path.name
            read_opener = gzip.open if src_path.suffix == ".gz" else open
            write_opener = gzip.open if dst_path.suffix == ".gz" else open
            with read_opener(src_path, "rt", encoding="utf-8") as src_handle:
                with write_opener(dst_path, "wt", encoding="utf-8") as dst_handle:
                    for line in src_handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        total_rows += 1
                        row_id = get_row_id(row)
                        if row_id is not None:
                            row_id_str = str(row_id)
                            replacement = retry_rows_by_id.get(row_id_str)
                            if replacement is not None:
                                row = replacement
                                replaced_ids.add(row_id_str)
                                replaced_rows += 1
                        dst_handle.write(json.dumps(row, ensure_ascii=False))
                        dst_handle.write("\n")

        not_found_in_base = expected_ids - replaced_ids
        if not_found_in_base:
            preview = sorted(not_found_in_base)[:5]
            raise RuntimeError(
                f"Could not find {len(not_found_in_base)} retried row(s) in base outputs for merge (examples: {preview})"
            )

        for src_path in source_files:
            staged_path = tmp_root / src_path.name
            try:
                staged_path.replace(src_path)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # Cross-device fallback for environments where the temp dir is
                # still on a different filesystem.
                shutil.move(str(staged_path), str(src_path))

    progress_logger.info(
        "Retry merge complete: replaced %d row(s) across %d file(s) in %s (total rows scanned: %d)",
        replaced_rows,
        len(source_files),
        output_dir,
        total_rows,
    )


def write_upload_normalized_jsonl(src_path: Path, dst_path: Path, drop_columns: set[str]) -> None:
    """Write a normalized copy for HF upload to avoid schema drift across rows/chunks."""
    read_opener = gzip.open if src_path.suffix == ".gz" else open
    write_opener = gzip.open if dst_path.suffix == ".gz" else open
    with read_opener(src_path, "rt", encoding="utf-8") as src_handle:
        with write_opener(dst_path, "wt", encoding="utf-8") as dst_handle:
            for line in src_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for column in drop_columns:
                    row.pop(column, None)
                dst_handle.write(json.dumps(row, ensure_ascii=False))
                dst_handle.write("\n")


def handle_checkpoint_cleanup_error(exc: OSError, checkpoints_dir: Path | None = None) -> bool:
    """Handle OSError from checkpoint cleanup, returning True if it was a non-fatal cleanup error.

    This handles the known NFS race condition where shutil.rmtree fails with ENOTEMPTY
    even though the directory appears empty (due to .nfs lock files or timing issues).
    """
    # Only handle "Directory not empty" errors
    if exc.errno != errno.ENOTEMPTY:
        return False

    error_filename = exc.filename or ""

    # If this is clearly a checkpoint-related path, treat it as a non-fatal error
    # The error typically occurs in paths like: .../checkpoints/00000
    if "checkpoint" in str(error_filename).lower():
        progress_logger.warning(
            "Checkpoint cleanup failed for %s (directory not empty). "
            "This is a known NFS race condition - treating as success.",
            error_filename,
        )
        if error_filename:
            try:
                shutil.rmtree(error_filename, ignore_errors=True)
            except Exception as cleanup_exc:
                progress_logger.warning("Failed to remove checkpoint dir %s: %s", error_filename, cleanup_exc)
        return True

    # If checkpoints_dir provided, check if error is within it
    if checkpoints_dir is not None and error_filename:
        try:
            error_path = Path(error_filename).resolve()
            checkpoints_root = checkpoints_dir.resolve()
            if error_path == checkpoints_root or checkpoints_root in error_path.parents:
                progress_logger.warning(
                    "Checkpoint cleanup failed for %s (directory not empty). Treating as success.",
                    error_path,
                )
                shutil.rmtree(error_path, ignore_errors=True)
                return True
        except Exception:
            pass

    return False


class RetryFailedFilter(PipelineStep):
    name = "Retry failed filter"
    type = "Filter"

    def __init__(self, retry_ids: set[str]):
        super().__init__()
        self.retry_ids = retry_ids

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            row = doc.metadata.get("row", {})
            row_id = get_row_id(row) or doc.id
            if row_id in self.retry_ids:
                yield doc


class SkipExistingFilter(PipelineStep):
    name = "Skip existing output filter"
    type = "Filter"

    def __init__(self, existing_ids: set[str]):
        super().__init__()
        self.existing_ids = existing_ids

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            row = doc.metadata.get("row", {})
            row_id = get_row_id(row) or doc.id
            if row_id in self.existing_ids:
                continue
            yield doc


def download_and_extract_longbench(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "longbench_data.zip"
    data_dir = cache_dir / "data"

    if data_dir.exists() and any(data_dir.iterdir()):
        logger.info("Using cached LongBench data from %s", data_dir)
        return data_dir

    logger.info("Downloading LongBench data from %s...", LONGBENCH_DATA_URL)
    urlretrieve(LONGBENCH_DATA_URL, zip_path)
    logger.info("Downloaded to %s", zip_path)

    logger.info("Extracting data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)
    logger.info("Extracted to %s", data_dir)

    zip_path.unlink()
    return data_dir


def sample_jsonl(source_path: Path, dest_path: Path, max_examples: int) -> Path:
    if dest_path.exists():
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as src, dest_path.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= max_examples:
                break
            if line.strip():
                dst.write(line)
    return dest_path


def ensure_ruler_jsonl(
    dataset_name: str,
    split: str,
    cache_dir: Path,
    max_examples: int | None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if max_examples:
        jsonl_path = cache_dir / f"{split}_sample_{max_examples}.jsonl"
        if jsonl_path.exists():
            return jsonl_path
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        dataset.to_json(str(jsonl_path))
        return jsonl_path

    jsonl_path = cache_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    dataset = load_dataset(dataset_name, split=split)
    dataset.to_json(str(jsonl_path))
    return jsonl_path


def run_datatrove_translation(
    input_path: Path,
    output_dir: Path,
    logging_dir: Path,
    checkpoints_dir: Path,
    language: str,
    fields_to_translate: list[str],
    vllm_url: str,
    model_name: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    enable_thinking: bool,
    request_timeout: float | None,
    max_concurrent_generations: int,
    max_concurrent_documents: int | None,
    records_per_chunk: int,
    retry_count: int,
    output_mode: str,
    language_key: str,
    include_source_language: bool,
    include_lang_fields: bool,
    include_failed: bool,
    skip_completed: bool,
    clean_output: bool,
    progress_interval: int,
    progress_label: str,
    retry_ids: set[str] | None,
    translation_profile: str,
    save_failed_attempts: bool,
    failed_attempts_filename: str,
) -> None:
    if retry_ids is not None and clean_output:
        progress_logger.warning(
            "Retry mode requested with clean_output enabled; ignoring clean_output to preserve base outputs for merge."
        )
        clean_output = False

    if clean_output:
        progress_logger.info("Clean output enabled; removing prior outputs for %s", output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(checkpoints_dir, ignore_errors=True)
        shutil.rmtree(logging_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    retry_output_dir: Path | None = None
    writer_output_dir = output_dir
    if retry_ids is not None:
        retry_output_dir = output_dir / "__retry_tmp__"
        shutil.rmtree(retry_output_dir, ignore_errors=True)
        retry_output_dir.mkdir(parents=True, exist_ok=True)
        writer_output_dir = retry_output_dir
        progress_logger.info(
            "Retry mode: writing retry rows to temporary output dir %s before merging into %s",
            retry_output_dir,
            output_dir,
        )

    failed_attempts_path = logging_dir / failed_attempts_filename if save_failed_attempts else None
    configure_failed_translation_artifact(failed_attempts_path)
    if failed_attempts_path is not None:
        progress_logger.info("Saving failed translation attempts to %s", failed_attempts_path)
    if translation_profile != TRANSLATION_PROFILE_DEFAULT:
        progress_logger.info("Using translation profile: %s", translation_profile)

    existing_output_files = sorted(output_dir.glob("*.jsonl*"))
    checkpoint_files = [path for path in checkpoints_dir.rglob("*") if path.is_file()]
    existing_ids: set[str] | None = None
    if skip_completed:
        if existing_output_files:
            progress_logger.info(
                "Resume detected: found %d output file(s) in %s",
                len(existing_output_files),
                output_dir,
            )
        if checkpoint_files:
            progress_logger.info(
                "Resume detected: found %d checkpoint file(s) in %s",
                len(checkpoint_files),
                checkpoints_dir,
            )
        if existing_output_files and not checkpoint_files and retry_ids is None:
            existing_ids = collect_existing_row_ids(output_dir)
            if existing_ids:
                progress_logger.info("Resume: skipping %d rows already in output", len(existing_ids))
            else:
                progress_logger.info("Resume: no row ids found in existing outputs; skip-by-output disabled")
        elif not existing_output_files and not checkpoint_files:
            progress_logger.info("Resume: no existing outputs or checkpoints found")
    else:
        progress_logger.info("Resume disabled: --no-skip-completed set")

    reader = JsonlReader(
        data_folder=str(input_path.parent),
        glob_pattern=input_path.name,
        recursive=False,
        adapter=make_reader_adapter(fields_to_translate),
    )

    if retry_ids is not None:
        total_docs = len(retry_ids)
    else:
        total_docs = count_jsonl_rows(input_path) if progress_interval > 0 else None
        if total_docs is not None and existing_ids:
            total_docs = max(total_docs - len(existing_ids), 0)

    if language == "en":
        output_writer = ProgressJsonlWriter(
            output_folder=str(writer_output_dir),
            output_filename="${rank}.jsonl",
            compression="gzip",
            adapter=output_adapter,
            total_docs=total_docs,
            progress_interval=progress_interval,
            progress_label=progress_label,
        )
        pipeline = [reader]
        if retry_ids is not None:
            pipeline.append(RetryFailedFilter(retry_ids))
        if existing_ids:
            pipeline.append(SkipExistingFilter(existing_ids))
        pipeline.extend(
            [
                PassthroughTranslator(
                    target_language=language,
                    fields_to_translate=fields_to_translate,
                    output_mode=output_mode,
                    language_key=language_key,
                    include_source_language=include_source_language,
                    include_lang_fields=include_lang_fields,
                    include_failed=include_failed,
                ),
                output_writer,
            ]
        )
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=1,
            workers=1,
            logging_dir=str(logging_dir),
            skip_completed=skip_completed,
        )
        completed = False
        try:
            executor.run()
            completed = True
        except OSError as exc:
            if not handle_checkpoint_cleanup_error(exc, checkpoints_dir):
                raise
            progress_logger.info("Translation completed (checkpoint cleanup had a non-fatal error)")
            completed = True
        finally:
            # Clean up any leftover checkpoint files that may cause issues on NFS
            if checkpoints_dir.exists():
                try:
                    shutil.rmtree(checkpoints_dir, ignore_errors=True)
                except Exception:
                    pass
        if completed and retry_output_dir is not None:
            merge_retry_outputs_into_existing(output_dir, retry_output_dir, retry_ids)
            shutil.rmtree(retry_output_dir, ignore_errors=True)
        return

    if not model_name:
        raise ValueError("model_name must be set for non-English translation jobs")

    output_writer = ProgressJsonlWriter(
        output_folder=str(writer_output_dir),
        output_filename="${rank}_chunk_${chunk_index}.jsonl",
        compression="gzip",
        adapter=output_adapter,
        total_docs=total_docs,
        progress_interval=progress_interval,
        progress_label=progress_label,
    )

    config = InferenceConfig(
        server_type="endpoint",
        endpoint_url=normalize_vllm_endpoint(vllm_url),
        model_name_or_path=model_name,
        use_chat=True,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_documents=max_concurrent_documents,
        default_generation_params={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "enable_thinking": enable_thinking,
        },
        request_timeout=request_timeout,
    )

    rollout_fn = partial(
        translate_fields,
        target_language=language,
        fields_to_translate=fields_to_translate,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        enable_thinking=enable_thinking,
        retry_count=retry_count,
        output_mode=output_mode,
        language_key=language_key,
        include_source_language=include_source_language,
        include_lang_fields=include_lang_fields,
        include_failed=include_failed,
        translation_profile=translation_profile,
    )

    pipeline = [reader]
    if retry_ids is not None:
        pipeline.append(RetryFailedFilter(retry_ids))
    if existing_ids:
        pipeline.append(SkipExistingFilter(existing_ids))
    pipeline.append(
        InferenceRunner(
            rollout_fn=rollout_fn,
            config=config,
            output_writer=output_writer,
            checkpoints_local_dir=str(checkpoints_dir),
            records_per_chunk=records_per_chunk,
            metadata_key="translation_output",
        )
    )
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        workers=1,
        logging_dir=str(logging_dir),
        skip_completed=skip_completed,
    )
    completed = False
    try:
        executor.run()
        completed = True
    except OSError as exc:
        if not handle_checkpoint_cleanup_error(exc, checkpoints_dir):
            raise
        progress_logger.info("Translation completed (checkpoint cleanup had a non-fatal error)")
        completed = True
    finally:
        # Clean up any leftover checkpoint files that may cause issues on NFS
        if checkpoints_dir.exists():
            try:
                shutil.rmtree(checkpoints_dir, ignore_errors=True)
            except Exception:
                pass
    if completed and retry_output_dir is not None:
        merge_retry_outputs_into_existing(output_dir, retry_output_dir, retry_ids)
        shutil.rmtree(retry_output_dir, ignore_errors=True)


def load_output_dataset(output_dir: Path):
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No JSONL outputs found in {output_dir}")
    if not UPLOAD_DROP_COLUMNS:
        data_files = [str(path) for path in files]
        return load_dataset("json", data_files=data_files, split="train")

    logger.info(
        "Loading output dataset from %s (dropping upload-only columns: %s)",
        output_dir,
        ", ".join(sorted(UPLOAD_DROP_COLUMNS)),
    )
    with tempfile.TemporaryDirectory(prefix="datatrove_upload_norm_") as tmp_dir:
        normalized_dir = Path(tmp_dir)
        normalized_files: list[str] = []
        for src_path in files:
            dst_path = normalized_dir / src_path.name
            write_upload_normalized_jsonl(src_path, dst_path, UPLOAD_DROP_COLUMNS)
            normalized_files.append(str(dst_path))
        return load_dataset("json", data_files=normalized_files, split="train")


def upload_to_hub(
    output_dir: Path,
    hf_repo: str,
    config_name: str,
    hf_token: str,
    save_to_disk: bool,
    split_name: str,
) -> None:
    dataset = load_output_dataset(output_dir)
    if save_to_disk:
        dataset.save_to_disk(str(output_dir / "hf_dataset"))
    for attempt in range(HF_UPLOAD_RETRY_COUNT):
        try:
            dataset.push_to_hub(
                hf_repo,
                config_name=config_name,
                split=split_name,
                token=hf_token,
                private=False,
            )
            return
        except Exception as exc:
            error_text = str(exc).casefold()
            is_transient = any(marker in error_text for marker in HF_UPLOAD_TRANSIENT_ERROR_SUBSTRINGS)
            if attempt >= HF_UPLOAD_RETRY_COUNT - 1 or not is_transient:
                raise
            delay_seconds = HF_UPLOAD_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "HF upload failed for %s/%s (%s) with transient error; retrying in %ds (%d/%d): %s",
                hf_repo,
                config_name,
                split_name,
                delay_seconds,
                attempt + 1,
                HF_UPLOAD_RETRY_COUNT,
                exc,
            )
            time.sleep(delay_seconds)


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
