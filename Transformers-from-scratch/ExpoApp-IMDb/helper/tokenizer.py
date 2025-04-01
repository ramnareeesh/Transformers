import tensorflow as tf
import tensorflow_datasets as tfds
import sentencepiece as spm
import numpy as np


# Load IMDb dataset
def load_imdb_texts():
    dataset = tfds.load("imdb_reviews", as_supervised=True)
    train_data, test_data = dataset["train"], dataset["test"]

    train_texts = [text.numpy().decode("utf-8") for text, _ in train_data]
    test_texts = [text.numpy().decode("utf-8") for text, _ in test_data]

    return train_texts + test_texts


# Save dataset to text file (needed for training SentencePiece tokenizer)
def save_texts_to_file(texts, filename="imdb_texts.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")


# Train BPE tokenizer using SentencePiece
def train_bpe_tokenizer(data_file="imdb_texts.txt", vocab_size=1000, model_prefix="bpe_imdb"):
    spm.SentencePieceTrainer.train(
        input=data_file, model_prefix=f"{model_prefix}_{vocab_size}", vocab_size=vocab_size,
        model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    return f"{model_prefix}_{vocab_size}"


# Load trained BPE tokenizer
def load_bpe_tokenizer(model_prefix="bpe_imdb"):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


# Tokenize dataset using BPE tokenizer
def tokenize_with_bpe(texts, tokenizer, max_len=256):
    tokenized = [tokenizer.encode(text, out_type=int) for text in texts]
    tokenized = [seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in
                 tokenized]
    return np.array(tokenized)


def tokenize(vocab_size=1000):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO & WARNING messages

    # Load IMDb texts
    texts = load_imdb_texts()

    # Save texts and train BPE tokenizer
    save_texts_to_file(texts)

    tokenizer_prefix = train_bpe_tokenizer(vocab_size=vocab_size)
    print("Tokenizer_prefix:", tokenizer_prefix)
    tokenizer = load_bpe_tokenizer(tokenizer_prefix)

    return tokenizer

# tokenizer = tokenize(vocab_size=2000)
#
# print(tokenizer.encode("The cat sat on the mat", out_type=int))
# print(tokenizer.encode("The cat sat on the mat", out_type=str))
#
# # print(tokenize_with_bpe(["The cat sat on the mat"], tokenize(vocab_size=1000)))
# # print(tokenizer.decode([108, 22, 36, 9, 36, 68, 8, 16, 36]))