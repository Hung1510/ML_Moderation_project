# Vietnamese Regional Discrimination — project starter kit

Everything from session 1, organized. Read `DATA_README.md` for the dataset details
and the data-quality issues found in the original ViRDC release.

## Folder layout

```
virdc_project/
├── README.md                 <- you are here
├── DATA_README.md            <- dataset data card + the leakage/label findings
├── data/
│   ├── raw_tiktok/           <- ORIGINAL ViRDC Tiktok files (contaminated; reference only)
│   │   ├── train.csv  test.csv  val.csv  tiktok.csv
│   ├── clean/                <- USE THESE: deduplicated, leakage-free splits
│   │   ├── train_clean.csv  val_clean.csv  test_clean.csv
│   └── all_data/             <- larger preprocessed pool (numeric labels; alldata has a stray label 4)
│       ├── alldata.csv  balanced_data.csv
├── scripts/
│   └── clean_resplit_virdc.py  <- regenerates data/clean/ from data/raw_tiktok/
└── training/
    └── finetune_distilbert_uitvsfc.py  <- Week 1: your first fine-tuning run (paste into Colab)
```

## What to do, in order

1. **Week 1 — learn the pipeline.** Open `training/finetune_distilbert_uitvsfc.py`,
   paste it into Google Colab cell-by-cell (GPU runtime on). It trains on UIT-VSFC,
   NOT ViRDC — a clean solved task so you can confirm the mechanics work. Target
   macro-F1 ~0.85-0.92.

2. **Later (Phase 0 reproduction) — use the clean ViRDC splits.** Train baselines on
   `data/clean/train_clean.csv`, score on `data/clean/test_clean.csv`. Compare your
   Random Forest F1 to ViRDC's reported 0.97; the honest number will likely be lower,
   and that gap is a publishable result.

   Label mapping: `{khác: 0, phân biệt: 1, ủng hộ: 2}`. Real class balance is skewed
   (~3592 / 1151 / 303 in train) — report **macro-F1**, not accuracy.

3. To regenerate the clean splits yourself: from the project root, run
   `python scripts/clean_resplit_virdc.py`.

## Source
ViRDC dataset/paper: https://github.com/anhuynh219/bigdata  (arXiv:2411.02587)
