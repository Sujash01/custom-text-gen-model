import joblib
import re

# ── Load models ────────────────────────────────
vectorizer = joblib.load("models/vectorizer.pkl")
model_type = joblib.load("models/type_model.pkl")
model_tone = joblib.load("models/tone_model.pkl")


# ── RULE 1: Detect interrogation ───────────────
def is_question(text):
    question_words = [
        "who", "what", "when", "where", "why", "how",
        "is", "are", "do", "does", "did", "can", "could", "should"
    ]

    text = text.lower()

    # check for question mark
    if "?" in text:
        return True

    # check starting word
    first_word = text.split()[0] if text.split() else ""
    if first_word in question_words:
        return True

    return False


# ── RULE 2: Detect meaningless text ────────────
def is_garbage(text):
    text = text.strip()

    # too short
    if len(text) < 4:
        return True

    # no vowels → likely nonsense
    if not re.search(r"[aeiou]", text.lower()):
        return True

    # mostly random chars
    if len(text.split()) == 1 and len(text) > 6:
        return True

    return False


# ── MAIN PREDICTION ────────────────────────────
def predict(text):
    text_clean = text.lower().strip()

    # 🔥 UNKNOWN detection
    if is_garbage(text_clean):
        return "unknown", "unknown"

    # 🔥 INTERROGATION detection
    if is_question(text_clean):
        return "interrogation", "neutral"

    # ML prediction
    X = vectorizer.transform([text_clean])

    pred_type = model_type.predict(X)[0]
    pred_tone = model_tone.predict(X)[0]

    return pred_type, pred_tone


# ── CLI LOOP ───────────────────────────────────
if __name__ == "__main__":
    print("=== Advanced Text Classifier ===")

    while True:
        user_input = input("\nEnter text (or 'exit'): ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        t, tone = predict(user_input)

        print(f"\nPredicted Type : {t}")
        print(f"Predicted Tone : {tone}")