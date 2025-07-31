import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

model = tf.keras.models.load_model("imdb_rnn_model.h5")
word_index = imdb.get_word_index()

index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"

def review_to_sequence(review, word_index, max_len=200):
    words = review.lower().split()
    sequence = [word_index.get(word, 2) for word in words]
    return pad_sequences([sequence], maxlen=max_len)

st.title("🎬 IMDb Review Sentiment Analysis")
review = st.text_area("Enter a movie review:")
if st.button("Predict Sentiment"):
    if review.strip() != "":
        sequence = review_to_sequence(review, word_index)
        prediction = model.predict(sequence)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}** (Confidence: {prediction:.2f})")
    else:
        st.warning("Please enter a valid review.")
