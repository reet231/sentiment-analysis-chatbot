import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

print("Program Started")

# Load dataset
try:
    data = pd.read_csv("Data.csv")
    print("CSV Loaded Successfully!")
    print(data.head())
except Exception as e:
    print("Error loading CSV:", e)
    exit()

# Check required columns
if "text" not in data.columns or "sentiment" not in data.columns:
    print("Error: CSV must contain 'text' and 'sentiment' columns.")
    exit()

# Input and output
X = data["text"]
y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Data Split Successfully!")

# Create model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training Model...")
model.fit(X_train, y_train)
print("Model Training Completed!")

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "sentiment_model.pkl")

print("Model Saved Successfully!")