import streamlit as st
from preprocessing import preprocess
from train import *

st.title("Transformers: Encoder Only Task")
st.header("IMDb Sentiment Analysis")

st.write("---")
col1, col2 = st.columns(2)
with col1:
    vocab_size = st.number_input("Enter the vocab size: ", value=2000, min_value=1000, max_value=10000, step=1000)

with col2:
    max_len = st.selectbox("Select max sequence length: ", [256, 512])

if st.button("Preprocess data"):
    st.write("---")
    with st.spinner("Preprocessing data..."):
        st.session_state["return_dict"] = preprocess(vocab_size, max_len)  # Store in session_state

    st.write(st.session_state["return_dict"]["shapes dict"])

st.write("---")

col_model_1, col_model_2 = st.columns(2)
with col_model_1:
    num_layers = st.number_input("No. of layers", value=3, min_value=1, max_value=6, step=1)
    d_model = st.selectbox("Embedding dimension", [256, 128, 512])

with col_model_2:
    n_heads = st.number_input("No. of attention heads", min_value=1, value=8, max_value=8, step=1)
    d_ff = st.selectbox("Dense layer dimension", [1024, 512, 2048])
    dropout_rate = st.number_input("Droupout rate", value=0.1, min_value=0.05, step=0.05)

if st.button("Instantiate model") and "return_dict" in st.session_state:
    train = Train(num_layers, d_model, n_heads, d_ff, dropout_rate, vocab_size, max_len)
    st.write(train.model.get_hyperparameters())

else:
    st.warning("Please preprocess the data first.")

if st.button("Train Model") and "return_dict" in st.session_state:
    st.write("---")

    train_dataset, val_dataset = st.session_state["return_dict"]["train_dataset"], st.session_state["return_dict"][
        "val_dataset"]
    epochs = st.number_input("Enter number of epochs", value=20, min_value=1, max_value=50, step=1)

    # UI elements
    st_progress = st.progress(0)
    st_text = st.empty()  # Placeholder for text updates

    # Initialize and train model
    train = Train(num_layers, d_model, n_heads, d_ff, dropout_rate, vocab_size, max_len)
    history = train.train_model(train_dataset, val_dataset, epochs, st_progress, st_text)

    # Display final results
    st.success("Training Completed!")
    st.line_chart({"Train Loss": history["loss"], "Validation Loss": history["val_loss"]})
    st.line_chart({"Train Accuracy": history["binary_accuracy"], "Validation Accuracy": history["val_binary_accuracy"]})

# model training



