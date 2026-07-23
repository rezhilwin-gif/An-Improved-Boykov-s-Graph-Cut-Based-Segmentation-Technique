# IBGC-CRF-SPSST

An improved **Boykov Graph Cut** + **DenseCRF** cell-nucleus segmentation
pipeline for the Herlev Pap-smear dataset, using a VGG16-FCN8 network to
produce the foreground probability map that seeds the graph cut.

Pipeline: `VGG16-FCN8 (foreground probability)` → `SLIC superpixel smoothing`
→ `Boykov graph cut` → `DenseCRF boundary refinement`.

## Project structure

```
IBGC-CRF-SPSST/
├── Data/                      # <- paste the Herlev dataset class folders in here
│   ├── normal_superficiel/
│   ├── normal_intermediate/
│   ├── normal_columnar/
│   ├── light_dysplastic/
│   ├── moderate_dysplastic/
│   ├── severe_dysplastic/
│   └── carcinoma_in_situ/
├── checkpoints/                # trained model weights, split, sweep config, logs
├── graphs/                     # sanity-check / stage-visualization figures
├── outputs/                    # final aggregate metrics (json/csv/plots)
├── src/
│   ├── config.py                # paths, seed, device, class list, GT color map
│   ├── dataset.py                # GT parsing, bias correction, crop, Dataset class
│   ├── model.py                  # VGG16-FCN8 architecture + inference helper
│   ├── postprocess.py            # superpixels, graph cut, CRF, run_full_pipeline
│   ├── inspect_data.py           # environment + dataset + GT color sanity checks
│   ├── train.py                  # trains the FCN head, saves checkpoints
│   ├── tune_hyperparams.py       # sweeps graph-cut/CRF hyperparameters
│   ├── final_eval.py             # full held-out evaluation, metrics + plots
│   └── visualize_preds.py        # 4-panel stage visualizations
├── run_pipeline.py             # runs every stage end to end
├── requirements.txt
├── environment.yml
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate ibgc-crf-spsst
```

`pydensecrf` and `PyMaxflow` are hard requirements — `src/postprocess.py`
fails loudly at import time if either is missing, rather than silently
degrading the graph cut / CRF steps to stubs.

## Add the dataset

Copy the Herlev Pap-smear class folders directly into `Data/`, so that you
end up with e.g. `Data/normal_superficiel/*.BMP`, `Data/light_dysplastic/*-d.bmp`,
etc. `src/dataset.py::scan_dataset()` pairs each `*-d.bmp` ground-truth mask
with its matching raw `.BMP` image.

## Run

Run the whole pipeline end to end:

```bash
python run_pipeline.py
```

Or run each stage individually:

```bash
python src/inspect_data.py        # environment + dataset + GT color sanity checks -> graphs/
python src/train.py               # trains VGG16-FCN8 -> checkpoints/best_model.pt
python src/tune_hyperparams.py    # sweeps graph-cut/CRF params -> checkpoints/best_sweep_config.json
python src/final_eval.py          # held-out evaluation -> outputs/final_metrics.json, .csv, bar chart
python src/visualize_preds.py     # 4-panel stage figures -> graphs/pred_sample_*.png
```

`run_pipeline.py --only inspect,train` runs a chosen subset of stages;
`run_pipeline.py --skip-train` reuses an existing `checkpoints/best_model.pt`.

## Outputs

- `checkpoints/best_model.pt` — best VGG16-FCN8 weights (by validation Dice)
- `checkpoints/training_config.json` — winning LR/epoch config + val Dice
- `checkpoints/train_val_split.json` — the exact train/val split used, reused by tuning + eval
- `checkpoints/best_sweep_config.json` — best graph-cut λ/σ, SLIC segments, CRF iterations
- `checkpoints/sweep_log.csv` — every hyperparameter combination tried
- `graphs/gt_color_check.png` — verifies the dataset's ground-truth color scheme
- `graphs/pred_sample_*.png` — preprocessed image / foreground probability / graph cut / CRF mask
- `outputs/final_metrics.json` / `.csv` — aggregate + per-image accuracy, precision, recall, specificity
- `outputs/final_metrics_bar.png` — bar chart of the aggregate metrics

## Notes

- `boykov_graph_cut()` inverts PyMaxflow's raw `get_grid_segments()` output
  (see the docstring in `src/postprocess.py`) — without the inversion, label
  `1` ends up meaning background, not foreground/nucleus.
- Ground truth is parsed by nearest-color classification (red=background,
  dark blue=cytoplasm, light blue=nucleus), not grayscale thresholding.
- The image crop and its ground-truth mask always share one bounding box
  (`dataset.py::load_pair`), used identically at train and inference time.
- The pipeline was first sanity-checked on a single sample, confirming the full model → graph cut → CRF flow runs end to end and produces a plausible nucleus mask.
