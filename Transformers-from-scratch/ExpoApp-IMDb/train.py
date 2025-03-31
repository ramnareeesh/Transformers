from model.encoder import TransformerEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


class Train():

    def __init__(
            self,
            num_layers=3,
            d_model=256,
            num_heads=8,
            d_ff=1024, dropout=0.1,
            vocab_size=2000,
            max_length=256
    ):
        self.model = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,  # Matching BPE vocab size
            max_len=max_length,
            dropout_rate=dropout
        )

    def print_model_summary(self, layer=2):
        return self.model.enc_layers[layer].build_graph().summary()

    def get_hyperparameters(self):
        return self.model.get_hyperparameters()

    def train_model(self, train_dataset, val_dataset, epochs, st_progress, st_text):
        history = {"loss": [], "val_loss": [], "binary_accuracy": [], "val_binary_accuracy": []}

        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=BinaryCrossentropy(label_smoothing=0.1, from_logits=False),
            metrics=[BinaryAccuracy()]
        )

        for epoch in range(epochs):
            hist = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=1,  # Train one epoch at a time
                batch_size=32,
                # verbose=0  # Suppress default output
            )

            # Collect history
            history["loss"].append(hist.history["loss"][0])
            history["val_loss"].append(hist.history["val_loss"][0])
            history["binary_accuracy"].append(hist.history["binary_accuracy"][0])
            history["val_binary_accuracy"].append(hist.history["val_binary_accuracy"][0])

            # Update Streamlit UI
            st_progress.progress((epoch + 1) / epochs)  # Update progress bar
            st_text.write(
                f"Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.4f}, Accuracy: {history['binary_accuracy'][-1]:.4f}")

        return history








