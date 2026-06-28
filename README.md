# wildfire-drone

YOLO fire segmentation → 100×100 grid → ConvLSTM wildfire spread forecast.

## Mac setup

```bash
git clone https://github.com/malihashar/wildfire-drone.git
cd wildfire-drone
chmod +x setup_mac.sh upload_to_kaggle.sh
./setup_mac.sh
source venv/bin/activate
```

### Copy assets from Windows

These folders are gitignored — copy them manually (USB, AirDrop, or zip):

```
models/convlstm/          # best_model.pt, latest_model.pt
models/yolo/              # fire_smoke_seg_best.pt
dataset/simulations/      # sim_*.pt
dataset/normalization.json
```

On Windows before you leave:

```powershell
Compress-Archive -Path models, dataset -DestinationPath wildfire-assets.zip
```

On Mac:

```bash
unzip ~/Downloads/wildfire-assets.zip -d .
```

### Kaggle API (Mac)

```bash
mkdir -p ~/.kaggle
# paste kaggle.json from kaggle.com/settings
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle kagglehub
```

Upload simulation batches:

```bash
./upload_to_kaggle.sh
```

GPU training stays on Kaggle (`kaggle_train.ipynb`). Local Mac uses **MPS** on Apple Silicon automatically when you pass `--device auto`.

### Run the pipeline

```bash
source venv/bin/activate

python scripts/predict_wildfire.py path/to/image.jpg \
  --sim dataset/simulations/sim_000.pt \
  --device auto
```

See [docs/yolo_integration.md](docs/yolo_integration.md) for the full YOLO → ConvLSTM flow.
