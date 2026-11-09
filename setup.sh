#!/usr/bin/env bash
# One-time setup: venv + libraries + pip freeze (task A1)
set -e
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib
pip freeze > requirements.txt
