import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parents[3]),str(HERE)]
from candidate import install
install()
import pytest
raise SystemExit(pytest.main(['-q',str(HERE/'test_storage_contracts.py'),str(HERE/'test_pyfront_boolean_storage.py')]))
