# import libraries
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# download nltk corpus (first time only)
import nltk

nltk.download('all')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample text data and labels (use more data for better performance)
texts = [
    "This is an amazing product", "I love this phone", "The product is bad", "Worst purchase ever",
    "great experience", "horrible service", "I like this", "I dislike this", "amazing quality", "b
ad quality"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 for Positive, 0 for Negative sentiment

# Initialize the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure they all have the same length
X = pad_sequences(sequences, padding='post')

# Labels (binary sentiment classification)
y = np.array(labels)  # Ensure labels are in numpy array format

# Build the neural network model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))  # Embedding layer to convert words to vectors
model.add(SpatialDropout1D(0.2))  # Dropout to avoid overfitting
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer for sequential data
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model (with more epochs and batch size suitable for the dataset)
model.fit(X, y, epochs=5, batch_size=2)

# Test sentence for prediction
test_sentence = ["very bad class"]
test_seq = tokenizer.texts_to_sequences(test_sentence)
test_seq = pad_sequences(test_seq, padding='post', maxlen=X.shape[1])  # Ensure the shape is correct

# Make the prediction
prediction = model.predict(test_seq)

# Apply threshold to classify the sentiment (0.5 as the threshold)
if prediction >= 0.5:
    print(f"Predicted sentiment for '{test_sentence[0]}': Positive")
else:
    print(f"Predicted sentiment for '{test_sentence[0]}': Negative")

     
