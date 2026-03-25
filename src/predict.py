import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 1. Load models ────────────────────────────────────────
with open("models/type_model.pkl", "rb") as f:
    type_model = pickle.load(f)

with open("models/tone_model.pkl", "rb") as f:
    tone_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ── 2. Predict function ───────────────────────────────────
def predict(text):
    X = vectorizer.transform([text])
    predicted_type = type_model.predict(X)[0]
    predicted_tone = tone_model.predict(X)[0]
    return predicted_type, predicted_tone

# ── 3. Test samples ───────────────────────────────────────
tests = [
    "why did the chicken cross the road to get to the other side",
    "believe in yourself and you will achieve great things",
    "the ocean was calm and the sky was painted orange",
    "oh sure because everything always goes perfectly",
    "machine learning allows computers to learn from data",
    "hello"
]

print("=== PREDICTIONS ===\n")
for text in tests:
    t, tone = predict(text)
    print(f"Text : {text}")
    print(f"Type : {t} | Tone: {tone}")
    print()