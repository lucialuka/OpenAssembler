"""Run ordinary tests against isolated modules without editing production."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from candidate import install
install()
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
