import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from plot_training_data import plot_training_history

batch_size = 32  # Choose an appropriate batch size for training

# Read the dataset
with open("./calling-training-dataset.txt", "r") as file:
    conversations = file.readlines()

# Parse conversations
conversation_turns = [line.split(": ")[1].strip() for line in conversations if ": " in line]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(conversation_turns)
sequences = tokenizer.texts_to_sequences(conversation_turns)

# Generate training sequences
max_len = 20  # Choose an appropriate max length for sequences
train_data = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        train_data.append(n_gram_sequence)

# Pad sequences
train_data = pad_sequences(train_data, maxlen=max_len, padding='pre')

# Create input and output for the model
X = train_data[:, :-1]
y = train_data[:, -1]

# Split data into training and validation sets
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Create TensorFlow Dataset objects for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
print(len(tokenizer.word_index))
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32),
    tf.keras.layers.LSTM(16),
    # tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.LSTM(32),
    # tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model with validation data
history = model.fit(train_dataset, epochs=4, validation_data=val_dataset)

# Plot training history
plot_training_history(history)

# Save the model
model.save("next_word_prediction_model.h5")
