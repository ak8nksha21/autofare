import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so "app" can be imported
sys.path.insert(0, str(ROOT / "src"))  # so "autofare" can be imported
