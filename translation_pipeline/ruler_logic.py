"""RULER-specific constants and parsing helpers."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_RULER_DATASET = "lighteval/RULER-32768-gemma3-instruct"
DEFAULT_RULER_SPLITS = ["niah_single_1", "qa_1", "qa_2"]
RULER_FIELDS = ["input", "outputs"]

TRANSLATION_PROFILE_RULER_NIAH = "ruler_niah"
TRANSLATION_PROFILE_RULER_QA = "ruler_qa"

RULER_QA_INSTRUCTION_TEXT = (
    "Answer the question based on the given documents. "
    "Only give me the answer and do not output any other words."
)
RULER_QA_QUESTION_LABEL = "Question:"
RULER_QA_ANSWER_LABEL = "Answer:"
RULER_QA_LABEL_TRANSLATIONS = {
    "en": {
        RULER_QA_QUESTION_LABEL: RULER_QA_QUESTION_LABEL,
        RULER_QA_ANSWER_LABEL: RULER_QA_ANSWER_LABEL,
    },
    "de": {
        RULER_QA_QUESTION_LABEL: "Frage:",
        RULER_QA_ANSWER_LABEL: "Antwort:",
    },
    "es": {
        RULER_QA_QUESTION_LABEL: "Pregunta:",
        RULER_QA_ANSWER_LABEL: "Respuesta:",
    },
    "fr": {
        RULER_QA_QUESTION_LABEL: "Question:",
        RULER_QA_ANSWER_LABEL: "Reponse:",
    },
    "it": {
        RULER_QA_QUESTION_LABEL: "Domanda:",
        RULER_QA_ANSWER_LABEL: "Risposta:",
    },
    "pt": {
        RULER_QA_QUESTION_LABEL: "Pergunta:",
        RULER_QA_ANSWER_LABEL: "Resposta:",
    },
    "pl": {
        RULER_QA_QUESTION_LABEL: "Pytanie:",
        RULER_QA_ANSWER_LABEL: "Odpowiedz:",
    },
    "nl": {
        RULER_QA_QUESTION_LABEL: "Vraag:",
        RULER_QA_ANSWER_LABEL: "Antwoord:",
    },
}
RULER_NIAH_INTRO_PREFIX = "A special magic number is hidden within the following text."
RULER_NIAH_QUESTION_PREFIX = "What is the special magic number for "
RULER_NIAH_QUESTION_SUFFIX = " mentioned in the provided text is"
RULER_NIAH_MIN_BODY_LINES = 64
DIGIT_SEQUENCE_RE = re.compile(r"\d+")


def is_ruler_niah_profile(translation_profile: str | None) -> bool:
    return (translation_profile or "").strip().lower() == TRANSLATION_PROFILE_RULER_NIAH


def is_ruler_qa_profile(translation_profile: str | None) -> bool:
    return (translation_profile or "").strip().lower() == TRANSLATION_PROFILE_RULER_QA


def get_ruler_qa_label_translation(target_language: str, english_label: str) -> str | None:
    return RULER_QA_LABEL_TRANSLATIONS.get(target_language.lower(), {}).get(english_label)


def split_outer_whitespace(text: str) -> tuple[str, str, str]:
    start = 0
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[:start], text[start:end], text[end:]


def is_numeric_only_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.isdigit()


def are_digit_sequences_preserved(source_text: str, translated_text: str) -> bool:
    source_digit_sequences = DIGIT_SEQUENCE_RE.findall(source_text)
    if not source_digit_sequences:
        return True
    return all(seq in translated_text for seq in source_digit_sequences)


def extract_ruler_qa_input_parts(text: str) -> dict[str, str] | None:
    if not text:
        return None

    instruction = RULER_QA_INSTRUCTION_TEXT
    tail_instruction_idx = text.rfind(instruction)
    if tail_instruction_idx < 0:
        return None

    question_idx = text.find(RULER_QA_QUESTION_LABEL, tail_instruction_idx + len(instruction))
    if question_idx < 0:
        return None

    answer_idx = text.rfind(RULER_QA_ANSWER_LABEL)
    if answer_idx < 0 or answer_idx <= question_idx:
        return None

    # We only support the canonical RULER QA layout where the text ends at the
    # answer label (plus optional whitespace).
    if text[answer_idx + len(RULER_QA_ANSWER_LABEL):].strip():
        return None

    has_head_instruction = text.startswith(instruction)
    head_instruction = instruction if has_head_instruction else ""
    body_start = len(instruction) if has_head_instruction else 0
    if tail_instruction_idx < body_start:
        return None

    return {
        "head_instruction": head_instruction,
        "body": text[body_start:tail_instruction_idx],
        "tail_instruction": instruction,
        "between_tail_instruction_and_question": text[tail_instruction_idx + len(instruction):question_idx],
        "question_label": text[question_idx:question_idx + len(RULER_QA_QUESTION_LABEL)],
        "question_text": text[question_idx + len(RULER_QA_QUESTION_LABEL):answer_idx],
        "answer_label_and_suffix": text[answer_idx:],
    }


def extract_ruler_niah_input_parts(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    lines = text.splitlines()
    if len(lines) < (RULER_NIAH_MIN_BODY_LINES + 2):
        return None

    intro_line = lines[0]
    question_line = lines[-1]
    body_lines = lines[1:-1]

    if not intro_line.startswith(RULER_NIAH_INTRO_PREFIX):
        return None
    if not question_line.startswith(RULER_NIAH_QUESTION_PREFIX):
        return None
    if not question_line.endswith(RULER_NIAH_QUESTION_SUFFIX):
        return None
    if not body_lines:
        return None

    unique_body_lines = set(body_lines)
    # niah_single_1 is highly repetitive with a hidden needle line embedded inside.
    # Keep the parser conservative so it only triggers on the intended layout.
    if len(unique_body_lines) > 8:
        return None

    return {
        "intro_line": intro_line,
        "body_lines": body_lines,
        "question_line": question_line,
        "line_count": len(lines),
        "body_unique_line_count": len(unique_body_lines),
    }
