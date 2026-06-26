import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()