# Sign Language Recognition using Spatial Attention-Based 3D GCN

Implementation attempt of the paper "Spatial Attention-Based 3D Graph Convolutional Neural Network for Sign Language Recognition" (Al-Hammadi et al., Sensors 2022) on the KArSL dataset.

**Status:** Incomplete due to hardware limitations. Only tested on small subset (10 signs out of 502).

---

## Paper Reference

Al-Hammadi, M., et al. "Spatial Attention-Based 3D Graph Convolutional Neural Network for Sign Language Recognition." Sensors 2022, 22, 4558. https://doi.org/10.3390/s22124558

---

## Dataset

This project uses the KArSL-502 dataset (502 Arabic sign language classes).

**To obtain the dataset, contact the authors at:** https://hamzah-luqman.github.io/KArSL/

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Setup Dataset

Organize the KArSL dataset so it has 3 signer folders, each with train/test splits:
```
KArSL/
├── 01/
│   ├── train/
│   └── test/
├── 02/
│   ├── train/
│   └── test/
└── 03/
    ├── train/
    └── test/
```

You also need the `KARSL-502_Labels.xlsx` file from the dataset authors.

### 2. Data Preparation

This combines the train/test folders from all 3 signers, removes leading zeros from folder names, and creates CSV files mapping each video path to its label.

```bash
python data_preparation.py \
  --base-path "path/to/KArSL" \
  --labels-file "path/to/KARSL-502_Labels.xlsx" \
  --csv-dir "csvs"
```

This produces:
- `csvs/train.csv` — all training videos with columns: `file_path`, `Label`, `SignID`
- `csvs/test.csv` — all test videos with the same columns

### 3. Filter Dataset (Optional)

To train on a smaller subset (e.g. only the first 10 signs):
```bash
python filter.py --input csvs/train.csv --output csvs/train_10.csv --max-signid 10
python filter.py --input csvs/test.csv --output csvs/test_10.csv --max-signid 10
```

This produces:
- `csvs/train_10.csv` — filtered to signs with SignID <= 10
- `csvs/test_10.csv` — filtered to signs with SignID <= 10

### 4. Training

This processes videos with MediaPipe (extracts 25 body/hand landmarks per frame), builds graph data, and trains the GCN model. All preprocessed data and the trained model are saved to the `--save-dir`.

```bash
python train.py \
  --train-csv csvs/train_10.csv \
  --test-csv csvs/test_10.csv \
  --save-dir Saved_10 \
  --epochs 100 \
  --batch-size 4 \
  --hidden-channels 8 \
  --num-layers 3 \
  --num-heads 2
```

This produces in `Saved_10/`:
- `train_data.pth` — preprocessed training graphs (so you can skip video processing on re-runs)
- `test_data.pth` — preprocessed test graphs
- `label_encoder.pkl` — sklearn LabelEncoder mapping class indices to Arabic sign labels
- `gcn_model.pth` — trained model weights (best test accuracy checkpoint)

On subsequent runs, `train.py` will detect existing `.pth` files and offer to skip preprocessing.

### 5. Inference

Using Python:
```python
from inference import predict_video
from model import load_model
import pickle

model = load_model('Saved_10/gcn_model.pth', in_channels=3, hidden_channels=8, out_channels=10, num_layers=3, num_heads=2)

with open('Saved_10/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

prediction = predict_video("path/to/video.mp4", model, label_encoder)
```

Or use the interactive notebook:
```bash
jupyter notebook playground.ipynb
```

---

## Requirements

See `requirements.txt` for dependencies.

---

## Limitations

- Hardware constraints prevented training on full dataset
- Only tested on 10 signs with reduced model capacity
- Results may not match paper performance

---

## Acknowledgments

- Paper authors for the architecture
- KArSL dataset creators