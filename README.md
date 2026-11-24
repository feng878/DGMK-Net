# M2KT / DGMK-Net

PyTorch implementation of **DGMK-Net** for radiology report generation, built on
top of the R2Gen-style encoder-decoder framework and evaluated on MIMIC-CXR and
IU X-ray.

DGMK-Net introduces three main contributions on top of a standard
visual-encoder + Transformer decoder backbone:

1. **MSVFA** — Multi-Scale Visual Feature Adaptive aggregation. Features from
   several stages of the visual extractor are fused with a learned scale
   attention, producing richer visual representations than a single-stage
   extractor (`modules/multi_scale_extractor.py`).
2. **DD-GAT** — Disease-Dependency Graph Attention Network. Disease labels are
   modelled as graph nodes with label co-occurrence edges; graph attention
   layers propagate disease-aware context into the decoder
   (`modules/graph_attention.py`).
3. **SACL** — Symptom-Aware Contrastive Learning. An InfoNCE-style bidirectional
   contrastive loss with Jaccard-distance-based adaptive margins pulls together
   reports with similar symptom profiles (`modules/contrastive_loss.py`).

An auxiliary label loss (BCE over the 14 CheXpert findings) is jointly
optimised with the captioning loss.

## Repository structure

```
main_dgmk.py               # Single entry point: train/eval DGMK-Net
config/                    # yacs-based YAML configs (_BASE_ inheritance)
models/                    # DGMK-Net (full model + single-scale ablation)
modules/                   # Dataloaders, losses, metrics, graph attention, etc.
scripts/                   # Smoke-test synthetic dataset generator
misc/                      # Logging utilities
```

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `torchvision`, `numpy`, `pandas`,
`Pillow`, `tqdm`, `pyyaml`, `yacs`, `tensorboardX`, `pycocoevalcap`.

## Data preparation

The configs expect the datasets under a `data/` directory (git-ignored):

```
data/
├── iu_xray/
│   ├── images/                       # IU X-ray images
│   ├── annotation.json               # R2Gen-style annotation file
│   └── r2gen_split/id_label.csv      # Image-to-label mapping
└── mimic/
    ├── files/                        # MIMIC-CXR-JPG images
    ├── mimic-cxr-2.0.0-chexpert.csv  # CheXpert labeler output
    └── finding/chexpert_labeled.csv  # Labelled finding subset
```

* **MIMIC-CXR** requires credentialed access via [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).
  Run the official CheXpert labeler to obtain `mimic-cxr-2.0.0-chexpert.csv`.
* **IU X-ray** (a.k.a. Open-I) is available from the
  [OpenI](https://openi.nlm.nih.gov/faq) collection; the annotation/split
  format follows R2Gen.

## Training

```bash
# MIMIC-CXR
python main_dgmk.py --cfg config/mimic_dgmk.yml --gpu 0

# IU X-ray
python main_dgmk.py --cfg config/iu_dgmk.yml --gpu 0
```

Useful options:

| Option | Description |
| --- | --- |
| `--cfg` | YAML config file (supports `_BASE_` inheritance) |
| `--set_cfgs KEY VALUE ...` | Override config keys from the command line |
| `--expe_name` | Suffix appended to the experiment directory name |
| `--resume` | Path to a checkpoint directory to resume from |
| `--gpu` | GPU id (via `CUDA_VISIBLE_DEVICES`) |
| `--use_amp` | Enable mixed-precision training |

Setting `no_multi_scale: True` switches to the single-scale ablation model
(`models/dgmk_model_single_scale.py`).

Training features: AMP mixed precision, gradient accumulation (`grad_accum_steps`),
NaN/Inf guard, pluggable LR scheduler (`StepLR` by default, `WarmupStepLR`
available via `lr_scheduler`), early stopping on a monitored metric (default
`CIDEr`), and fixed random seeds for reproducibility.

## Smoke test

To verify the installation without downloading the real datasets, generate a
tiny synthetic IU X-ray dataset and run one epoch:

```bash
python scripts/make_iu_smoke_data.py
python main_dgmk.py --cfg config/iu_dgmk.yml --gpu 0 --epochs 1 \
    --batch_size 2 --num_workers 0 --expe_name smoke_test
```

This exercises the full pipeline end-to-end (data loading, vocabulary
construction, DGMK-Net forward/backward, beam-search sampling, NLG metric
computation, checkpointing and CSV logging) on 16 synthetic studies
(8 train / 4 val / 4 test). The synthetic data and all outputs are
git-ignored.

## Evaluation

Metrics are computed automatically after each validation/test epoch:

* **NLG metrics** (via `pycocoevalcap`): BLEU-1..4, METEOR, ROUGE-L, CIDEr.

These NLG metrics are logged, plotted to TensorBoard and written to the
`records/` CSV summaries.

Generated reports are dumped as JSON under the experiment directory in
`results_dgmk/` (git-ignored).

## Results and records

Per-dataset best-metric CSV summaries are written to `records/` at the end of
training (created automatically; git-ignored).

## License

This project is released under the [MIT License](LICENSE).
