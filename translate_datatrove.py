#!/usr/bin/env python3
"""Compatibility wrapper for the refactored translation pipeline."""

from translation_pipeline.logic import *
from translation_pipeline.pipeline_ops import *
from translation_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
