#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python scripts/generate_demo_data.py
python scripts/prepare_data.py \
  data/raw/demo_metar.csv
python scripts/train_models.py \
  data/processed/hourly_training_data.csv \
  --torch-epochs 5
python -m pytest -q
