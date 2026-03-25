import joblib
import pandas as pd
from datasets import load_dataset

# ── 1. Load trained models ─────────────────────
vectorizer = joblib.load("models/vectorizer.pkl")
model_type = joblib.load("models/type_model.pkl")
model_tone = joblib.load("models/tone_model.pkl")

# ── 2. Load FineWeb (streaming) ────────────────
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True
)

print("Streaming FineWeb...")

results = []

# ── 3. Predict on 2000 samples ─────────────────
for i, sample in enumerate(dataset):

    text = sample.get("text", "")

    # skip empty
    if not text or len(text.strip()) < 10:
        continue

    # basic preprocess
    text_clean = text.lower().strip()

    # vectorize
    X = vectorizer.transform([text_clean])

    # predict
    pred_type = model_type.predict(X)[0]
    pred_tone = model_tone.predict(X)[0]

    results.append({
        "text": text[:200],   # truncate for readability
        "type": pred_type,
        "tone": pred_tone
    })

    # progress print
    if i % 200 == 0:
        print(f"Processed: {i}")

    # limit
    if i >= 2000:
        break

# ── 4. Save results ────────────────────────────
df = pd.DataFrame(results)
df.to_csv("dataset/fineweb_predictions.csv", index=False)

print("\nDone! Saved to dataset/fineweb_predictions.csv")