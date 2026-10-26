"""End-to-end demo: run with `make demo` or `python examples/run_demo.py`."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedserve.demo import run_demo

if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
