import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

#Load the data
data = pd.read_csv('src/Dico.csv')
words = data['Word']
genders = data['Gender']

#encode the gender
label_encoder = LabelEncoder()
genders_encoded = label_encoder.fit_transform(genders)

#split between train and test
train_words, test_words, train_labels, test_labels = train_test_split(
    words, genders_encoded, test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_words)
train_sequences = tokenizer.texts_to_sequences(train_words)
test_sequences = tokenizer.texts_to_sequences(test_words)

max_word_length = max(len(seq) for seq in train_sequences)
train_padded = pad_sequences(train_sequences, maxlen=max_word_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_word_length, padding='post')

# Build the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
hidden_units = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_word_length))
model.add(LSTM(hidden_units, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=10, batch_size=128, validation_data=(test_padded, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_padded, test_labels)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Predict the gender of a new word
new_word = 'covid'
new_word_sequence = tokenizer.texts_to_sequences([new_word])
new_word_padded = pad_sequences(new_word_sequence, maxlen=max_word_length, padding='post')
prediction = model.predict(new_word_padded)
gender = 'M' if prediction > 0.5 else 'F'
print(f'The predicted gender of "{new_word}" is {gender}')