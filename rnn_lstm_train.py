import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text dataset
data = """deep learning is fun
deep learning is powerful
deep learning is amazing"""

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

# Padding
max_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 10, input_length=max_len-1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train
model.fit(X, y, epochs=200, verbose=1)

# Save model
model.save("next_word_lstm.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save max_len
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("Training completed and model saved!")