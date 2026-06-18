import os
import pickle
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = os.path.join("Saved_models", "sentiment_model.h5")
TOKENIZER_PATH = os.path.join("Saved_models", "tokenizer.pickle")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

def extract_title_from_url(url: str) -> str | None:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            if soup.title and soup.title.string:
                return soup.title.string.strip()
    except requests.RequestException:
        pass
    return None

def analyze_sentiment(headline: str, max_len: int = 120) -> int:
    sequences = tokenizer.texts_to_sequences([headline])
    padded_seqs = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_seqs)
    return int(np.argmax(prediction))
