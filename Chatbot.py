import joblib
import random
from utils import clean_text

# Load model
model = joblib.load("sentiment_model.pkl")

RESPONSES = {
    "positive": [
        "😊 That's great to hear! Thank you for your kind words.",
        "😊 I'm so glad things are going well! How else can I help?",
        "😊 Wonderful! Thanks for sharing that with us.",
        "😊 Really happy to hear that! Let us know if you need anything.",
    ],
    "negative": [
        "😔 I'm sorry to hear that. Let me help resolve this for you.",
        "😔 I understand your frustration. Can you share more details?",
        "😔 I apologize for the inconvenience. I'll do my best to fix this.",
        "😔 That's not acceptable. Let me look into this right away.",
    ],
    "neutral": [
        "🙂 Got it! How can I assist you further?",
        "🙂 Thank you for reaching out. What would you like to know?",
        "🙂 Sure, I can help with that. Please go ahead.",
        "🙂 Of course! Let me know how I can help you today.",
    ]
}

def get_sentiment_and_confidence(message):
    cleaned = clean_text(message)
    sentiment = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    max_idx = list(classes).index(sentiment)
    confidence = round(proba[max_idx] * 100, 1)
    return sentiment, confidence

def generate_response(sentiment):
    return random.choice(RESPONSES.get(sentiment, RESPONSES["neutral"]))

print("=" * 40)
print("   Sentiment Chatbot Started!")
print("   Type 'quit' to exit")
print("=" * 40)
print()

while True:
    user_message = input("You: ").strip()

    if not user_message:
        continue

    if user_message.lower() == "quit":
        print("Bot: Goodbye! Have a great day! 👋")
        break

    sentiment, confidence = get_sentiment_and_confidence(user_message)
    response = generate_response(sentiment)

    print(f"Detected Sentiment : {sentiment} ({confidence}% confidence)")
    print(f"Bot                : {response}")
    print()