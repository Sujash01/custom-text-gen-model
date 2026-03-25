import pandas as pd

# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv("dataset/dataset.csv")

print("=== RAW DATA ===")
print(f"Shape: {df.shape}")
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head(3)}")

# ── 2. Strip whitespace ───────────────────────────────────
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = df[col].str.strip()

# ── 3. Lowercase ──────────────────────────────────────────
for col in df.columns:
    df[col] = df[col].str.lower()

# ── 4. Fix inconsistent source names ─────────────────────
df['source'] = df['source'].replace('jokes_dataset', 'joke_dataset')

# ── 5. Fix mismatched source ──────────────────────────────
mask = (df['source'] == 'joke_dataset') & (df['type'] != 'joke')
df.loc[mask, 'source'] = 'quotes_dataset'

# ── 6. Remove exact duplicates ────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=['text'])
print(f"\nRemoved {before - len(df)} exact duplicate(s). Rows now: {len(df)}")

# ── 7. Consolidate similar source names ───────────────────
df['source'] = df['source'].replace({
    'wikipedia_refined': 'wikipedia',
    'generated_refined': 'generated',
    'self_help_blog':    'self_help'
})

print(f"\nCleaned source values:\n{df['source'].value_counts()}")

# ── 8. Save ───────────────────────────────────────────────
df.to_csv("dataset/dataset_cleaned.csv", index=False)
print("\n=== DONE! Saved as dataset/dataset_cleaned.csv ===")