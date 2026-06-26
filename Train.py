import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from utils import clean_text

print("Program Started")

# Load dataset
try:
    data = pd.read_csv("Data.csv")
    print("CSV Loaded Successfully!")
    print(data.head())
    data["text"] = data["text"].apply(clean_text)
    print("Text Cleaned Successfully!")
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
    random_state=42,
    stratify=y        # makes sure all 3 classes are in train and test
)

print("Data Split Successfully!")

# Upgraded model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),       # captures "not good", "very happy" etc
        max_features=10000,
        sublinear_tf=True         # reduces effect of very common words
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced"   # handles unequal class sizes
    ))
])

print("Training Model...")
model.fit(X_train, y_train)
print("Model Training Completed!")

# Test model
predictions = model.predict(X_test)

# Basic accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Detailed report
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# Confusion matrix
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "sentiment_model.pkl")
print("\nModel Saved Successfully!")