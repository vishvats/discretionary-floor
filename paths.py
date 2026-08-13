"""Repository paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data"
SPAN = RAW
PANEL = ROOT / "panel"
FIGS = ROOT / "figures"
VIX = RAW / "india_vix.csv"
