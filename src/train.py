import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# ── 1. Load cleaned data ──────────────────────────────────
df = pd.read_csv("dataset/dataset_cleaned.csv")
df = df.dropna()

print(f"Training on {len(df)} samples")

# ── 2. Vectorize text ─────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

# ── 3. Train TYPE model ───────────────────────────────────
y_type = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=42)

type_model = LogisticRegression(max_iter=1000)
type_model.fit(X_train, y_train)

print("\n=== TYPE MODEL ===")
print(classification_report(y_test, type_model.predict(X_test)))

# ── 4. Train TONE model ───────────────────────────────────
y_tone = df['tone']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_tone, test_size=0.2, random_state=42)

tone_model = LogisticRegression(max_iter=1000)
tone_model.fit(X_train2, y_train2)

print("\n=== TONE MODEL ===")
print(classification_report(y_test2, tone_model.predict(X_test2)))

# ── 5. Save models ────────────────────────────────────────
with open("models/type_model.pkl", "wb") as f:
    pickle.dump(type_model, f)

with open("models/tone_model.pkl", "wb") as f:
    pickle.dump(tone_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n=== DONE! Models saved to models/ ===")