from att_view import MV , HV , NV
import streamlit as st
st.title("Model View Attention Visualization")
input_text=st.text_area("Enter text:")
if st.button("Visualize Attention"):
    st.components.v1.html(MV(input_text).data, height=1000, scrolling=True)
    # st.components.v1.html(HV(input_text).data, height=1000, scrolling=True)
    # st.components.v1.html(NV(input_text).data, height=1000, scrolling=True)