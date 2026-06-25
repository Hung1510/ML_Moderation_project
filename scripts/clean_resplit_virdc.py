"""
clean_resplit_virdc.py
----------------------
Turns the contaminated official ViRDC Tiktok split into a clean, leakage-free
train/val/test split. Run once; commit the outputs; use them for every model
you benchmark so comparisons are fair.

WHY THIS EXISTS (read the DATA_README): the official split has ~52% duplicate
rows in train and ~29% of test sentences copied verbatim from train. That
inflates scores. We fix it by deduplicating to unique sentences FIRST, then
splitting.

Usage:
    python clean_resplit_virdc.py
Edit IN_DIR / OUT_DIR below if your files live elsewhere.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------
IN_DIR  = "data/raw_tiktok"   # run this script from the project root          # folder holding train.csv / test.csv / val.csv
OUT_DIR = "data/clean"               # where the *_clean.csv files get written
SEED    = 42                # fixed seed => the split is reproducible by anyone
LABEL_MAP = {"khác": 0, "phân biệt": 1, "ủng hộ": 2}  # string -> int

# ----------------------------------------------------------------------------
# 1. Pool the three official files back into one table.
#    The official train/test/val boundaries are what we DON'T trust, so we
#    dissolve them and rebuild from scratch.
# ----------------------------------------------------------------------------
parts = [pd.read_csv(f"{IN_DIR}/{name}.csv") for name in ("train", "test", "val")]
df = pd.concat(parts, ignore_index=True)
df["sentence"] = df["sentence"].astype(str).str.strip()
print(f"[1] pooled rows:                 {len(df):>6}")
print(f"    unique sentences:            {df['sentence'].nunique():>6}")

# ----------------------------------------------------------------------------
# 2. Resolve label conflicts. A handful of identical sentences were tagged with
#    two different labels by the annotators. We keep the MAJORITY label per
#    sentence; if there's an exact tie we drop the sentence (too ambiguous to
#    trust). This collapses each unique sentence to exactly one (sentence,label)
#    row, which also guarantees no sentence can leak across splits later.
# ----------------------------------------------------------------------------
def majority_label(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None                      # tie -> mark for dropping
    return counts.idxmax()               # the most frequent label

resolved = df.groupby("sentence")["label"].apply(majority_label)
dropped_ties = resolved.isna().sum()
resolved = resolved.dropna().reset_index()       # columns: sentence, label
print(f"[2] sentences after dedup:       {len(resolved):>6}  (dropped {dropped_ties} tied)")

# ----------------------------------------------------------------------------
# 3. Map the Vietnamese string labels to integers. Models want numbers, not
#    text, and fixing the mapping here keeps class IDs identical everywhere.
# ----------------------------------------------------------------------------
resolved["label"] = resolved["label"].map(LABEL_MAP)
resolved = resolved.dropna(subset=["label"])     # drop anything unmapped (safety)
resolved["label"] = resolved["label"].astype(int)

# ----------------------------------------------------------------------------
# 4. Stratified split 70 / 15 / 15.
#    "Stratified" = each split keeps the same class proportions as the whole,
#    so test isn't accidentally missing a class. We split twice: first peel off
#    30% for (val+test), then halve that into val and test.
# ----------------------------------------------------------------------------
train_df, temp_df = train_test_split(
    resolved, test_size=0.30, random_state=SEED, stratify=resolved["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
)

# ----------------------------------------------------------------------------
# 5. Verify the leak is actually gone. This assertion is the whole point — if
#    it ever fails, do not trust the numbers you get downstream.
# ----------------------------------------------------------------------------
s_tr, s_va, s_te = set(train_df.sentence), set(val_df.sentence), set(test_df.sentence)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "LEAK!"
print(f"[5] overlap train/val/test:           0  (verified leakage-free)")

# ----------------------------------------------------------------------------
# 6. Save. Also print the class balance so you can see what you're training on.
# ----------------------------------------------------------------------------
for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
    part.to_csv(f"{OUT_DIR}/{name}_clean.csv", index=False)
    dist = part["label"].value_counts().sort_index().to_dict()
    print(f"[6] {name:<5} {len(part):>5} rows  class counts {dist}")

print("\nDone. Use train_clean.csv / val_clean.csv / test_clean.csv from here on.")
