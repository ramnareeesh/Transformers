from helper.tokenizer import tokenize, tokenize_with_bpe, load_bpe_tokenizer
from helper.dataset import load_and_train_test_split, train_val_split
import tensorflow.data as tfdata
def preprocess(vocab_size, max_len):
    tokenizer = tokenize(vocab_size=vocab_size)
    # tokenizer = load_bpe_tokenizer("bpe_model_2000")
    train_texts, train_labels, test_texts, test_labels = load_and_train_test_split()

    train_texts_tokenized = tokenize_with_bpe(train_texts, tokenizer, max_len=max_len)
    test_x = tokenize_with_bpe(test_texts, tokenizer, max_len=max_len)

    train_x, val_x, train_y, val_y = train_val_split(train_texts_tokenized, train_labels)

    # print("Train_X shape: ", train_x.shape)
    # print("Train_Y shape: ", train_y.shape)
    # print("Sample: ", train_x[0])
    # print()
    # print("Sample: ", train_x[1])
    # print()
    # print("Sample: ", train_x[2])
    # print("Val_X shape: ", val_x.shape)
    # print("Val_Y shape: ", val_y.shape)
    # print("Test_X shape: ", test_x.shape)
    # print("Test_Y shape: ", test_labels.shape)

    shapes = {
        "train_x": train_x.shape,
        "train_y": train_y.shape,
        "val_x": val_x.shape,
        "val_y": val_y.shape,
        "test_x": test_x.shape,
        "test_y": test_labels.shape
    }

    # Create TensorFlow Datasets
    train_dataset = tfdata.Dataset.from_tensor_slices((train_x, train_y)).batch(32)  # Create dataset and batch it
    val_dataset = tfdata.Dataset.from_tensor_slices((val_x, val_y)).batch(32)  # Create dataset and batch it

    return_dict = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "shapes dict": shapes,
        "test_x": test_x,
        "test_y": test_labels
    }

    return return_dict
if __name__ == '__main__':

    print(preprocess(vocab_size=3000, max_len=256)["shapes dict"])
