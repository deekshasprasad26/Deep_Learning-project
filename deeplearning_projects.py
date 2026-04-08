import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("next_word_lstm.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

st.title("Next Word Prediction (LSTM)")

input_text = st.text_input("Enter a sentence:")

if st.button("Predict"):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == np.argmax(predicted):
            output_word = word
            break

    st.success("Next word: " + output_word)
