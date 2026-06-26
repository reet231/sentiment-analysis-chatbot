import joblib

# Load model
model = joblib.load("sentiment_model.pkl")

def generate_response(sentiment):

    if sentiment == "positive":
        return (
            "😊 I'm happy to hear that! "
            "Thank you for your positive feedback."
        )

    elif sentiment == "negative":
        return (
            "😔 I'm sorry you're facing this issue. "
            "Let me help resolve it."
        )

    else:
        return (
            "🙂 Thank you for your message. "
            "How can I assist you further?"
        )

print("Sentiment Chatbot Started!")
print("Type 'quit' to exit.\n")

while True:

    user_message = input("You: ")

    if user_message.lower() == "quit":
        break

    sentiment = model.predict([user_message])[0]

    response = generate_response(sentiment)

    print("Detected Sentiment:", sentiment)
    print("Bot:", response)
    print()