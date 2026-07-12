"""
conftest.py

Pytest automatically discovers and runs this file before any tests in
this folder - no import needed elsewhere. Its job here is simple: add
src/ to Python's module search path, so test files can import
data_preprocessing, train, and evaluate the same way train.py and
evaluate.py already import from each other (e.g.
`from data_preprocessing import load_config`).

Without this, pytest imports test files from the project root, not
from inside src/ - so those imports would fail with
ModuleNotFoundError even though the exact same imports work fine when
running `python src/train.py` directly (Python auto-adds a script's
own folder to the search path when run directly, but not when pytest
collects it as a test module).
"""

import sys
import os

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_PATH))