# ViRDC dataset — data card (what's here, what to use, what's broken)

Source: https://github.com/anhuynh219/bigdata (the ViRDC paper repo, arXiv:2411.02587)
There is also a Hugging Face mirror at `anhuynh19/pbvm` (not downloaded here; couldn't
reach huggingface.co from this environment — worth checking, it may be cleaner).

## Files

| File | Rows (incl. header) | Columns | Labels | Text state |
|---|---|---|---|---|
| `Tiktok/train.csv` | 10,486 | `sentence`, `label` | string: `khác` / `phân biệt` / `ủng hộ` | word-segmented |
| `Tiktok/test.csv` | 3,496 | `sentence`, `label` | same 3 strings | word-segmented |
| `Tiktok/val.csv` | 3,496 | `sentence`, `label` | same 3 strings | word-segmented |
| `Tiktok/tiktok.csv` | 8,736 | (raw crawl) | — | — |
| `AllData/alldata.csv` | 20,493 | `processed_comment`, `label` | numeric 0/1/2/**4** | preprocessed |
| `AllData/balanced_data.csv` | 10,207 | `processed_comment`, `label` | numeric 0/1/2 | preprocessed |

The **Tiktok 3-way split (train+test+val ≈ 17K)** is the "17K ViRDC dataset" referenced
in your proposal. That's the canonical artifact to benchmark against for comparability.

## Label scheme (the README is misleading — use this)

The three classes are:
- `khác` = 0 = **Other** (non-discriminatory)
- `phân biệt` = 1 = **Discrimination** (regionally discriminatory)
- `ủng hộ` = 2 = **Supportive** (counter-speech / defending a region)

The repo README labels class 2 as "Discrimination against Middle region people" and
class 1 as "against Northern people." That is **wrong / incoherent** — the actual string
labels are `khác`/`phân biệt`/`ủng hộ`, which are other / discriminatory / supportive.
Don't trust the README's parenthetical descriptions; trust the string labels.

## ⚠️ Problems you need to know about (these affect your contributions)

### 1. The official train/test/val split is contaminated. Badly.
Measured directly on the files in `Tiktok/`:
- `train.csv`: **5,483 of 10,485 rows are exact duplicates** (~52%).
- **1,017 sentences in train also appear verbatim in test.**
- 998 train sentences also appear in val; 654 test sentences also appear in val.
- ~29% of test sentences (1,017 / 3,495) are literally copied from train.

This is sentence-level train/test leakage, not just keyword leakage. It almost certainly
inflates the reported Random Forest F1 of 0.97. **This strengthens your Phase 1 thesis** —
you don't even need adversarial perturbation to show their number is fragile; a clean,
de-duplicated, leakage-free re-split will likely drop RF substantially on its own. Do that
*first* as a sanity baseline, then layer adversarial perturbation on top.

### 2. The text is already preprocessed (word-segmented + cleaned).
Notice tokens like `tổ_quốc`, `chiến_tranh` — the underscores mean VnCoreNLP-style word
segmentation has already been applied, and stopwords look partly removed. ~56% of train
rows contain `_`.

**This is a real obstacle for your robustness contribution.** Your char-CNN + adversarial
(Telex/leetspeak/tone-removal) angle needs the *raw* surface form of comments. Segmented,
cleaned text has lost the original obfuscation. Options:
- Use `Tiktok/tiktok.csv` (the raw crawl) and re-derive labels by matching — fiddly.
- Check the Hugging Face mirror `anhuynh19/pbvm` for a raw version.
- Generate adversarial variants synthetically from the cleaned text anyway (you control the
  perturbation, so you don't strictly need the originals to *be* obfuscated — you inject it).
  This is probably your cleanest path and keeps the rule-based generator you already planned.

### 3. A stray label `4` in `alldata.csv` (84 rows).
`alldata.csv` contains labels {0, 1, 2, 4} — there is no class 3, and 4 is undefined. It's
noise/a labeling slip. `balanced_data.csv` already drops it. Use `balanced_data.csv` or
filter `alldata.csv` to {0,1,2}.

## Recommendation for Phase 0 / Phase 1
1. For your *training-wheels* DistilBERT run, you don't need this dataset — UIT-VSFC is fine
   and avoids these landmines while you learn the pipeline.
2. When you move to reproducing ViRDC: load `Tiktok/train.csv`+`test.csv`, then
   **de-duplicate and re-split with no overlap** before training anything. Report the
   leakage numbers above in your paper — that's a legitimate, citable critique of prior work.
3. Map string labels → ints consistently: `{'khác':0, 'phân biệt':1, 'ủng hộ':2}`.
