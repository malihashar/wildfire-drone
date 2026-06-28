#!/usr/bin/env bash
# One-time Mac setup for wildfire-drone.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Install Python 3 first: brew install python" >&2
  exit 1
fi

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt

echo ""
echo "Checking model assets (copy from Windows if missing):"
for path in \
  models/convlstm/best_model.pt \
  models/yolo/fire_smoke_seg_best.pt \
  dataset/normalization.json
do
  if [[ -e "${path}" ]]; then
    echo "  ok  ${path}"
  else
    echo "  MISSING  ${path}"
  fi
done

sim_count="$(find dataset/simulations -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
echo "  sims  ${sim_count} files in dataset/simulations/"

if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Kaggle credentials: ok (~/.kaggle/kaggle.json)"
else
  echo "Kaggle credentials: missing — add ~/.kaggle/kaggle.json"
fi

echo ""
echo "Running tests..."
python -m unittest discover -s tests -p "test_*.py" -q

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source venv/bin/activate"
echo ""
echo "Run inference:"
echo "  python scripts/predict_wildfire.py IMAGE.jpg --sim dataset/simulations/sim_000.pt"
