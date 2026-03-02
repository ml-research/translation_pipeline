"""LongBench-specific constants and helpers."""

from __future__ import annotations

LONGBENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

RENAMED_SUBJECTS = {
    "multifieldqa_en": "multifieldqa",
    "passage_retrieval_en": "passage_retrieval",
}

LONGBENCH_DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
LONGBENCH_FIELDS = ["input", "context", "answers"]


def canonical_subject_name(subject: str) -> str:
    return RENAMED_SUBJECTS.get(subject, subject)


def subject_matches_filter(subject: str, filter_items: list[str]) -> bool:
    subject_lower = subject.lower()
    canonical_lower = canonical_subject_name(subject).lower()
    return (
        subject_lower in filter_items
        or canonical_lower in filter_items
        or any(f in subject_lower or f in canonical_lower for f in filter_items)
    )
