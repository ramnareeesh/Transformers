from Transformer_model import TransformerModel
from pickle import load
from tensorflow import Module
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose

# Define the model parameters
h = 8 # Number of self-attention heads
d_k = 64 # Dimensionality of the linearly projected queries and keys
d_v = 64 # Dimensionality of the linearly projected values
d_model = 512 # Dimensionality of model layers' outputs
d_ff = 2048 # Dimensionality of the inner fully connected layer
n = 6 # Number of layers in the encoder stack

# Define the dataset parameters
enc_seq_length = 7 # Encoder sequence length
dec_seq_length = 12 # Decoder sequence length
enc_vocab_size = 2404 # Encoder vocabulary size
dec_vocab_size = 3864 # Decoder vocabulary size

# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)


class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)

        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)

        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)

        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        all_enc_attention = None
        all_dec_self_attention = None
        all_dec_enc_attention = None

        for i in range(dec_seq_length):
            # Predict an output token
            prediction,enc_attention, dec_self_attention, dec_enc_attention = self.transformer(encoder_input,
                                                                                               transpose(decoder_output.stack()), training=False)

            if i == 0:
                all_enc_attention = enc_attention

            # Always update decoder attentions as they change with each new token
            all_dec_self_attention = dec_self_attention
            all_dec_enc_attention = dec_enc_attention
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]

            # Write the selected prediction to the output array at the next
            # available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()
        output_str = []

        # Decode the predicted tokens into an output string
        for i in range(output.shape[0]):
            key = output[i]
            output_str.append(dec_tokenizer.index_word[key])

        return output_str, all_enc_attention, all_dec_self_attention, all_dec_enc_attention


# Sentence to translate
sentence = ['cat sat on the mat']

# Load the trained model's weights at the specified epoch
inferencing_model.load_weights('weights/wghts20.ckpt')

# Create a new instance of the 'Translate' class
translator = Translate(inferencing_model)

print(translator(sentence))

