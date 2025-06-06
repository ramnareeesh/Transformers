from .layers import Layer, MultiHeadAttention, AddNormalization, FeedForward, Dropout, Dense, PositionEmbeddingFixedWeights
from keras import Model, Input
from tensorflow.keras import layers

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, max_len, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model // num_heads, d_model // num_heads, d_model)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.add_norm2 = AddNormalization()
        self.dropout = Dropout(dropout_rate)

    def build_graph(self):
        input_layer = Input(shape=(self.max_len, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None))

    def call(self, x, mask=None):
        # Multi-Head Self-Attention
        attn_output = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.add_norm1(x, attn_output)

        # Feed-Forward Network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.add_norm2(x, ff_output)

        return x


class TransformerEncoder(Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        # needed attributes ------------
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout_rate
        # ------------
        self.embedding = PositionEmbeddingFixedWeights(max_len, vocab_size, d_model)
        self.enc_layers = [TransformerEncoderLayer(d_model, max_len, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.final_dense = Dense(1, activation="sigmoid")  # Binary classification

    def get_hyperparameters(self):
        return {
            "No. of encoder layers": self.num_layers,
            "Dimension of the embeddings": self.d_model,
            "No. of attention heads": self.num_layers,
            "No. of neurons in feed-forward": self.d_ff,
            "Vocabulary size": self.vocab_size,
            "Max. sequence length": self.max_len,
            "Dropout rate": self.dropout
        }

    def call(self, x, mask=None):
        x = self.embedding(x)
        for enc_layer in self.enc_layers:
            x = enc_layer(x)
        x = self.global_avg_pool(x)  # Reduce sequence to a single vector
        return self.final_dense(x)
