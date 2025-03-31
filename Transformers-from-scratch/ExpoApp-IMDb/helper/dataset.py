import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

import numpy as np


def load_and_train_test_split():
    # Load IMDb dataset with raw text
    imdb_dataset = tfds.load("imdb_reviews", as_supervised=True)

    # Extract training and test data
    train_data = imdb_dataset["train"]
    test_data = imdb_dataset["test"]

    # Convert TFDS dataset to lists
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for text, label in train_data:
        train_texts.append(text.numpy().decode("utf-8"))  # Convert from Tensor to string
        train_labels.append(label.numpy())

    for text, label in test_data:
        test_texts.append(text.numpy().decode("utf-8"))
        test_labels.append(label.numpy())

    # Convert labels to tensors
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_texts, train_labels, test_texts, test_labels

def train_val_split(train_tokens_padded, train_labels):
    train_x, val_x, train_y, val_y = train_test_split(train_tokens_padded, train_labels, test_size=0.1, random_state=42)
    return train_x, val_x, train_y, val_y

