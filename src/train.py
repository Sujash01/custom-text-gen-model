import pandas as pd
import joblib
import os

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ── 1. Load YOUR dataset ───────────────────────
df_main = pd.read_csv("dataset/dataset_cleaned.csv")
df_main = df_main.dropna(subset=['text', 'type', 'tone'])

print(f"Your dataset size: {len(df_main)}")

# ── 2. Load HuggingFace emotion dataset ───────
dataset = load_dataset("dair-ai/emotion")

df_emotion = pd.DataFrame(dataset['train'])

# ── 3. Map labels → emotion ───────────────────
emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

df_emotion['emotion'] = df_emotion['label'].map(emotion_map)

# ── 4. Map emotion → tone ─────────────────────
tone_map = {
    "sadness": "emotional",
    "joy": "casual",
    "love": "emotional",
    "anger": "serious",
    "fear": "serious",
    "surprise": "casual"
}

df_emotion['tone'] = df_emotion['emotion'].map(tone_map)

# ── 5. Add TYPE column ────────────────────────
df_emotion['type'] = "conversational"

# ── 6. Keep only required columns ─────────────
df_emotion = df_emotion[['text', 'type', 'tone']]

print(f"Emotion dataset size: {len(df_emotion)}")

# ── 7. Merge datasets ─────────────────────────
df = pd.concat([df_main, df_emotion])
df = df.drop_duplicates(subset='text')
df = df.reset_index(drop=True)

print(f"Final dataset size: {len(df)}")

# ── 8. Vectorization (IMPROVED) ───────────────
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(df['text'])

y_type = df['type']
y_tone = df['tone']

# ── 9. Train-test split ───────────────────────
X_train, X_test, y_type_train, y_type_test, y_tone_train, y_tone_test = train_test_split(
    X, y_type, y_tone, test_size=0.2, random_state=42
)

# ── 10. Train models ──────────────────────────
type_model = LinearSVC()
type_model.fit(X_train, y_type_train)

tone_model = LinearSVC()
tone_model.fit(X_train, y_tone_train)

# ── 11. Evaluate ──────────────────────────────
print("\n=== TYPE MODEL ===")
print(classification_report(y_type_test, type_model.predict(X_test)))

print("\n=== TONE MODEL ===")
print(classification_report(y_tone_test, tone_model.predict(X_test)))

# ── 12. Save models ───────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(type_model, "models/type_model.pkl")
joblib.dump(tone_model, "models/tone_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n=== DONE! Models saved ===")