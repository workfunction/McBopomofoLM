"""Paths for the offline evaluation tools. Everything generated (models, venv, walk_cli, caches, outputs)
lives under EVAL_WORK (default ~/.cache/mcbpmflm-eval); run scripts from that directory."""
import os
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
TOOLS = str(Path(__file__).resolve().parent)
WORK = os.environ.get("EVAL_WORK", str(Path.home() / ".cache" / "mcbpmflm-eval"))
MODELS = os.path.join(WORK, "models")
D512_DIR = os.path.join(MODELS, "d512")           # Funmula/0c-sequence-d512 @ d512-pf512-20260926
SLOTH_DIR = os.path.join(MODELS, "sloth")         # Luigi/sloth-ime-models: slothe.pt (25M), gate/train scripts, vocab
Q35_DIR = os.path.join(MODELS, "q35")             # Luigi/sloth-ime-models: pred_q35_60m/
G2PW_MODEL = os.path.join(WORK, "g2pw-model", "G2PWModel")
WALK_CLI = os.path.join(WORK, "walk_cli")
DATA_TXT = os.path.join(REPO, "Source", "Data", "data.txt")   # `make data.txt` in Source/Data first
BUNDLE = os.path.join(REPO, "SlothE", "Bundle", "SlothE")
FX = os.path.join(REPO, "McBopomofoTests", "SlothEFixtures")
