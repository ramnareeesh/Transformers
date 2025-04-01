import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from Transformer_model import TransformerModel
from Prepare_dataset import PrepareDataset
from Transformer_inference import Translate
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape
from keras.optimizers import Adam
from keras.optimizers.schedules import LearningRateSchedule
from keras.metrics import Mean
from keras.losses import sparse_categorical_crossentropy
import pickle
import re


def visualize_attention(attention_weights, layer_name, head_idx=None):
    """
    Visualize attention weights as a heatmap

    Parameters:
    attention_weights: Attention weights tensor
    layer_name: Name of the layer (for title)
    head_idx: If specified, visualize only this attention head
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert from tensor to numpy if needed
    if isinstance(attention_weights, tf.Tensor):
        attention_weights = attention_weights.numpy()

    # If it's multi-head attention and a specific head is requested
    if head_idx is not None and len(attention_weights.shape) > 3:
        weights = attention_weights[0, head_idx]
        plt.title(f"{layer_name} - Head {head_idx}")
    else:
        # If we want to average across all heads
        if len(attention_weights.shape) > 3:
            weights = np.mean(attention_weights[0], axis=0)
            plt.title(f"{layer_name} - Average across all heads")
        else:
            weights = attention_weights[0]
            plt.title(layer_name)

    # Create heatmap
    im = ax.imshow(weights, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set_xlabel("Attention To")
    ax.set_ylabel("Attention From")

    return fig

    # Define learning rate scheduler (from your training code)
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        step_num = cast(step_num, float32)
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Function to determine the latest epoch from weight files
def get_latest_epoch():
    if not os.path.exists("weights"):
        os.makedirs("weights")
        return 0

    weight_files = [f for f in os.listdir("weights") if f.startswith("wghts") and f.endswith(".ckpt.index")]
    if not weight_files:
        return 0

    # Extract epoch numbers from filenames
    epoch_nums = []
    for file in weight_files:
        match = re.search(r'wghts(\d+)\.ckpt\.index', file)
        if match:
            epoch_nums.append(int(match.group(1)))

    return max(epoch_nums) if epoch_nums else 0

    # Loss function for training
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction[0], from_logits=True) * mask
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(mask)

    # Accuracy function for training
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included
    mask = math.logical_not(math.equal(target, 0))
    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction[0], axis=2))
    accuracy = math.logical_and(mask, accuracy)
    # Cast the True/False values to 32-bit-precision floating-point numbers
    mask = cast(mask, float32)
    accuracy = cast(accuracy, float32)
    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(mask)

# Plot training history
def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure both dictionaries have the same keys
    epochs = sorted(set(train_losses.keys()).intersection(val_losses.keys()))

    ax.plot(epochs, [train_losses[epoch] for epoch in epochs], 'b-', label='Training Loss')
    ax.plot(epochs, [val_losses[epoch] for epoch in epochs], 'r-', label='Validation Loss')

    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    return fig


def display_feature_page():
    """Display the feature page content with transformer visualization functionality."""

    # Initialize session state to store model and results
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'output_tokens' not in st.session_state:
        st.session_state.output_tokens = None
    if 'enc_attention' not in st.session_state:
        st.session_state.enc_attention = None
    if 'dec_self_attention' not in st.session_state:
        st.session_state.dec_self_attention = None
    if 'dec_enc_attention' not in st.session_state:
        st.session_state.dec_enc_attention = None
    if 'translation_done' not in st.session_state:
        st.session_state.translation_done = False

    # Define model parameters
    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_model = 512  # Dimensionality of model layers' outputs
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    n = 6  # Number of layers in the encoder stack

    # Define the dataset parameters
    enc_seq_length = 7  # Encoder sequence length
    dec_seq_length = 12  # Decoder sequence length
    enc_vocab_size = 2404  # Encoder vocabulary size
    dec_vocab_size = 3864  # Decoder vocabulary size

    # Define training parameters
    # Define training parameters
    dropout_rate = 0.1
    beta_1 = 0.9
    beta_2 = 0.98
    epsilon = 1e-9
    batch_size = 64

    # Mode selection for Feature page
    app_mode = st.sidebar.selectbox("Choose the mode", ["Inference Mode", "Training Mode"])

    if app_mode == "Inference Mode":
        st.title("Transformer Inference Visualizer")

        # Introduction
        st.markdown("""
        This mode allows you to see how the transformer model translates from English to German 
        and visualizes the internal attention mechanisms at work.
        """)

        # Weight selection
        weight_files = []
        if os.path.exists("weights"):
            weight_files = [f for f in os.listdir("weights") if f.startswith("wghts") and f.endswith(".ckpt.index")]
            weight_files = [f.replace(".index", "") for f in weight_files]
        else:
            st.warning("No weight checkpoints found. Please switch to Training Mode to train the model first.")
            return

        selected_weight = st.selectbox(
            "Select checkpoint to use:",
            weight_files,
            index=0
        )

        checkpoint_path = os.path.join("weights", selected_weight)

        # Input for translation
        st.header("Translate English to German")
        input_sentence = st.text_input("Enter an English sentence:", "cat sat on the mat")

        # Run inference when button is clicked
        if st.button("Translate and Visualize"):
            with st.spinner("Loading model and translating..."):
                # Step 1: Initialize the model
                st.subheader("Step 1: Initializing the model")
                st.text("Loading model parameters and weights...")

                # Initialize the model with the selected weights
                st.session_state.model = TransformerModel(
                    enc_vocab_size, dec_vocab_size,
                    enc_seq_length, dec_seq_length,
                    h, d_k, d_v, d_model, d_ff, n, 0
                )

                st.session_state.model.load_weights(checkpoint_path)
                translator = Translate(st.session_state.model)
                st.session_state.output_tokens, st.session_state.enc_attention, st.session_state.dec_self_attention, st.session_state.dec_enc_attention = translator(
                    [input_sentence])

                # Set flag to indicate translation is complete
                st.session_state.translation_done = True

                # Force a rerun to show results
                st.experimental_rerun()

        # Only display visualizations if translation has been done
        if st.session_state.translation_done:
            # Final translation
            st.header("Final Translation")
            output = ' '.join([t for t in st.session_state.output_tokens if t not in ['start', 'eos']])
            st.success(f"Translation: {output}")

            # Visualize Attention Weights
            st.header("Attention Visualization")

            # Create tabs for different attention visualizations
            tab1, tab2, tab3 = st.tabs(
                ["Encoder Self-Attention", "Decoder Self-Attention", "Decoder-Encoder Attention"])

            with tab1:
                st.subheader("Encoder Self-Attention")
                st.markdown("""
                This visualization shows how words in the input sentence attend to other words. 
                Brighter colors indicate stronger attention connections.
                """)

                # Layer selection
                layer_idx = st.selectbox(
                    "Select encoder layer:",
                    range(1, len(st.session_state.enc_attention) + 1),
                    format_func=lambda x: f"Layer {x}"
                ) - 1

                # Head selection
                num_heads = st.session_state.enc_attention[0].shape[1]
                head_options = ["Average all heads"] + [f"Head {i + 1}" for i in range(num_heads)]
                head_selection = st.selectbox("Select attention head:", head_options)

                if head_selection == "Average all heads":
                    head_idx = None
                else:
                    head_idx = int(head_selection.split(" ")[1]) - 1

                # Plot the attention weights
                fig = visualize_attention(
                    st.session_state.enc_attention[layer_idx],
                    f"Encoder Layer {layer_idx + 1} Attention",
                    head_idx
                )
                st.pyplot(fig)

            with tab2:
                st.subheader("Decoder Self-Attention")
                st.markdown("""
                This shows how each position in the output attends to previous positions.
                Due to the causal mask, each position can only attend to itself and previous positions.
                """)

                # Layer selection
                layer_idx = st.selectbox(
                    "Select decoder layer:",
                    range(1, len(st.session_state.dec_self_attention) + 1),
                    format_func=lambda x: f"Layer {x}"
                ) - 1

                # Head selection
                num_heads = st.session_state.dec_self_attention[0].shape[1]
                head_options = ["Average all heads"] + [f"Head {i + 1}" for i in range(num_heads)]
                head_selection_dec = st.selectbox("Select decoder attention head:", head_options, key="dec_head")

                if head_selection_dec == "Average all heads":
                    head_idx = None
                else:
                    head_idx = int(head_selection_dec.split(" ")[1]) - 1

                # Plot the attention weights
                fig = visualize_attention(
                    st.session_state.dec_self_attention[layer_idx],
                    f"Decoder Self-Attention Layer {layer_idx + 1}",
                    head_idx
                )
                st.pyplot(fig)

            with tab3:
                st.subheader("Decoder-Encoder Cross-Attention")
                st.markdown("""
                This visualization shows how each position in the output attends to positions in the input.
                It helps understand which input words influence each output word during translation.
                """)

                # Layer selection
                layer_idx = st.selectbox(
                    "Select decoder-encoder layer:",
                    range(1, len(st.session_state.dec_enc_attention) + 1),
                    format_func=lambda x: f"Layer {x}"
                ) - 1

                # Head selection
                num_heads = st.session_state.dec_enc_attention[0].shape[1]
                head_options = ["Average all heads"] + [f"Head {i + 1}" for i in range(num_heads)]
                head_selection_cross = st.selectbox("Select cross-attention head:", head_options, key="cross_head")

                if head_selection_cross == "Average all heads":
                    head_idx = None
                else:
                    head_idx = int(head_selection_cross.split(" ")[1]) - 1

                # Plot the attention weights
                fig = visualize_attention(
                    st.session_state.dec_enc_attention[layer_idx],
                    f"Decoder-Encoder Attention Layer {layer_idx + 1}",
                    head_idx
                )
                st.pyplot(fig)

    # Training Mode
    else:  # Training Mode
        st.title("Transformer Training Visualizer")

        # Introduction
        st.markdown("""
        This mode allows you to train the transformer model and visualize the learning process.
        The model will be trained on an English-German translation dataset.
        """)

        # Get latest epoch from weight files
        latest_epoch = get_latest_epoch()

        # Show the latest epoch information
        if latest_epoch > 0:
            st.info(
                f"Found existing model checkpoint at epoch {latest_epoch}. Training will continue from epoch {latest_epoch + 1}.")
        else:
            st.info("No existing checkpoints found. Training will start from epoch 1.")

        # Training parameters - only number of epochs is customizable
        st.header("Training Parameters")

        additional_epochs = st.slider("Number of Additional Epochs to Train",
                                      min_value=1,
                                      max_value=20,
                                      value=2)

        # Display fixed parameters
        st.subheader("Fixed Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Batch Size: {batch_size}")
            st.text(f"Dropout Rate: {dropout_rate}")
        with col2:
            st.text(f"Model Dimension: {d_model}")
            st.text(f"Number of Heads: {h}")

        # Dataset information
        st.header("Dataset Information")
        st.info("The model will be trained on an English-German translation dataset.")

        # Only allow training if the dataset file exists
        if os.path.exists('english-german-both.pkl'):
            if st.button("Start Training"):
                # Initialize training components
                progress_text = st.empty()
                progress_bar = st.progress(0)

                # Create placeholders for loss charts and attention visualization
                loss_chart_placeholder = st.empty()
                epoch_status = st.empty()

                # Initialize optimizer
                optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

                # Initialize metrics
                train_loss_metric = Mean(name='train_loss')
                train_accuracy_metric = Mean(name='train_accuracy')
                val_loss_metric = Mean(name='val_loss')

                # Prepare the dataset
                progress_text.text("Loading and preparing dataset...")
                dataset = PrepareDataset()
                trainX, trainY, valX, valY, train_orig, val_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset(
                    'english-german-both.pkl')

                # Prepare dataset batches
                train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
                train_dataset = train_dataset.batch(batch_size)

                val_dataset = data.Dataset.from_tensor_slices((valX, valY))
                val_dataset = val_dataset.batch(batch_size)

                # Initialize model
                progress_text.text("Initializing model...")
                training_model = TransformerModel(
                    enc_vocab_size, dec_vocab_size,
                    enc_seq_length, dec_seq_length,
                    h, d_k, d_v, d_model, d_ff, n, dropout_rate
                )

                # Create checkpoint
                ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
                ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

                # Load existing weights if available
                if latest_epoch > 0:
                    latest_weights = f"weights/wghts{latest_epoch}.ckpt"
                    progress_text.text(f"Loading weights from {latest_weights}...")
                    training_model.load_weights(latest_weights)

                    # Try to load existing loss history
                    try:
                        with open('./train_loss.pkl', 'rb') as file:
                            train_loss_dict = pickle.load(file)
                        with open('./val_loss.pkl', 'rb') as file:
                            val_loss_dict = pickle.load(file)
                    except:
                        train_loss_dict = {}
                        val_loss_dict = {}
                else:
                    train_loss_dict = {}
                    val_loss_dict = {}

                # Start training
                start_time = time.time()
                start_epoch = latest_epoch + 1
                total_epochs = start_epoch + additional_epochs - 1

                # Define train step function
                def train_step(encoder_input, decoder_input, decoder_output):
                    with GradientTape() as tape:
                        # Run the forward pass of the model to generate a prediction
                        prediction = training_model(encoder_input, decoder_input, training=True)
                        # Compute the training loss
                        loss = loss_fcn(decoder_output, prediction)
                        # Compute the training accuracy
                        accuracy = accuracy_fcn(decoder_output, prediction)
                    # Retrieve gradients of the trainable variables with respect to the training loss
                    gradients = tape.gradient(loss, training_model.trainable_weights)
                    # Update the values of the trainable variables by gradient descent
                    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
                    train_loss_metric(loss)
                    train_accuracy_metric(accuracy)
                    return loss

                # Training loop
                for epoch in range(start_epoch, start_epoch + additional_epochs):
                    # Reset metrics at the start of each epoch
                    train_loss_metric.reset_states()
                    train_accuracy_metric.reset_states()
                    val_loss_metric.reset_states()

                    epoch_status.text(f"Epoch {epoch}/{total_epochs}")
                    progress_text.text(f"Training epoch {epoch}...")

                    # Iterate over the dataset batches
                    total_batches = len(list(train_dataset))
                    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
                        # Define the encoder and decoder inputs, and the decoder output
                        encoder_input = train_batchX[:, 1:]
                        decoder_input = train_batchY[:, :-1]
                        decoder_output = train_batchY[:, 1:]

                        # Perform training step
                        batch_loss = train_step(encoder_input, decoder_input, decoder_output)

                        # Update progress bar
                        progress = ((epoch - start_epoch) * total_batches + step) / (
                                    additional_epochs * total_batches)
                        progress_bar.progress(progress)

                        # Update display periodically
                        if step % 10 == 0:
                            progress_text.text(
                                f"Epoch {epoch} - Batch {step}/{total_batches} - Loss: {batch_loss:.4f}")

                            # Store current loss values
                            train_loss_dict[epoch - 1 + step / total_batches] = train_loss_metric.result().numpy()

                            # Update loss chart if we have validation data
                            if val_loss_dict:
                                fig = plot_training_history(train_loss_dict, val_loss_dict)
                                loss_chart_placeholder.pyplot(fig)

                    # Run validation after each epoch
                    progress_text.text(f"Running validation for epoch {epoch}...")
                    for val_batchX, val_batchY in val_dataset:
                        # Define the encoder and decoder inputs, and the decoder output
                        encoder_input = val_batchX[:, 1:]
                        decoder_input = val_batchY[:, :-1]
                        decoder_output = val_batchY[:, 1:]

                        # Generate a prediction
                        prediction = training_model(encoder_input, decoder_input, training=False)

                        # Compute the validation loss
                        loss = loss_fcn(decoder_output, prediction)
                        val_loss_metric(loss)

                    # Store the epoch loss values
                    train_loss_dict[epoch] = train_loss_metric.result().numpy()
                    val_loss_dict[epoch] = val_loss_metric.result().numpy()

                    # Update loss chart
                    # fig = plot_training_history(train_loss_dict, val_loss_dict)
                    # loss_chart_placeholder.pyplot(fig)

                    # Print epoch results
                    epoch_status.text(
                        f"Epoch {epoch}/{total_epochs}: "
                        f"Train Loss {train_loss_metric.result():.4f}, "
                        f"Train Accuracy {train_accuracy_metric.result():.4f}, "
                        f"Val Loss {val_loss_metric.result():.4f}"
                    )

                    # Save model checkpoint
                    save_path = ckpt_manager.save()
                    training_model.save_weights(f"weights/wghts{epoch}.ckpt")
                    progress_text.text(f"Saved checkpoint for epoch {epoch}")

                # Save the loss dictionaries at the end of training
                with open('./train_loss.pkl', 'wb') as file:
                    pickle.dump(train_loss_dict, file)
                with open('./val_loss.pkl', 'wb') as file:
                    pickle.dump(val_loss_dict, file)

                # Training complete
                training_time = time.time() - start_time
                progress_bar.progress(1.0)
                progress_text.text(f"Training Complete! Total time: {training_time:.2f}s")

                # Show final results
                st.header("Training Results")
                st.success(
                    f"Model trained for {additional_epochs} epochs "
                    f"(total {total_epochs} epochs) with final training loss: "
                    f"{train_loss_metric.result():.4f} and validation loss: {val_loss_metric.result():.4f}"
                )

                # Display the final loss chart
                # st.subheader("Training and Validation Loss")
                # final_fig = plot_training_history(train_loss_dict, val_loss_dict)
                # st.pyplot(final_fig)

                st.info(
                    f"Model weights have been saved. You can now switch to Inference Mode "
                    f"and select the checkpoint 'wghts{total_epochs}.ckpt' to use the newly trained model."
                )

        else:
            st.error(
                "Dataset file 'english-german-both.pkl' not found. Please ensure the dataset is properly prepared."
            )

