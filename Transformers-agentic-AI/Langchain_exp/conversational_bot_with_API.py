import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="AI Conversation Agent", page_icon="🤖", layout="wide")

load_dotenv()

# App title and description
st.title("🤖 AI Conversation Agent")
st.markdown("""
This app uses Hugging Face's API to generate conversational responses.
The conversation history is maintained for context awareness.
""")

# Sidebar for model settings
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["google/flan-t5-large", "google/flan-t5-base", "tiiuae/falcon-7b-instruct"],
    index=0
)

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
max_length = st.sidebar.slider("Max Response Length", min_value=64, max_value=2048, value=1024, step=32)

# API key status indicator
HUGGING_FACE_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGING_FACE_API_KEY:
    st.sidebar.error("API key not found in environment variables! Please set HUGGING_FACE_API_KEY.")
else:
    st.sidebar.success("API key configured")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LangChain components
@st.cache_resource
def get_conversation_chain():
    # Create a LLM from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={
            "temperature": temperature,
            "max_length": max_length,
            "do_sample": True,  # Enable sampling for more diverse outputs
            "top_p": 0.9,  # Use top-p sampling
            "top_k": 50,  # Limit to top 50 tokens
        }
    )

    # Create a more focused template for the conversation
    template = """You are a helpful, friendly, and engaging AI assistant. 
Respond directly and naturally to the conversation. 
Provide informative and contextually appropriate responses.

Conversation History:
{history}

Current Request:
Human: {input}
AI Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    # Create a conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    # Create the conversation chain
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    return conversation


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to talk about?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if API key is configured
    if not HUGGING_FACE_API_KEY:
        with st.chat_message("assistant"):
            st.error("Please set HUGGING_FACE_API_KEY environment variable before running the app.")
    else:
        try:
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get the conversation chain
                    conversation = get_conversation_chain()

                    # Get the response
                    response = conversation.predict(input=prompt)

                    # Post-process the response to remove any remaining template artifacts
                    clean_response = response.split("AI Assistant:")[-1].strip()
                    st.markdown(clean_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"An error occurred: {str(e)}")
                st.info("This could be due to model limitations or connectivity problems.")

# Add a reset button
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# CSS to improve the appearance
st.markdown("""
<style>
.stChatMessage {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)