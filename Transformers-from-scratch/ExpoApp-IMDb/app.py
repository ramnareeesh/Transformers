import streamlit as st
from preprocessing import preprocess

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

        return_dict = preprocess(vocab_size, max_len)
    st.write(return_dict["shapes dict"])

