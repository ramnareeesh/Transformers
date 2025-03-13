import streamlit as st
import os
import tempfile
import torch
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Model imports
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    pipeline
)

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader
)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_TEMP = 0.7
DEFAULT_MAX_LENGTH = 128
DEFAULT_MODEL = "facebook/blenderbot-400M-distill"
SUPPORTED_FILE_TYPES = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader
}

# Set device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# App configuration
st.set_page_config(
    page_title="Enhanced BlenderBot Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-container {
        border-radius: 15px;
        background-color: #f5f7f9;
        padding: 20px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e1f5fe;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sources-section {
        font-size: 0.8em;
        border-top: 1px solid #ddd;
        margin-top: 10px;
        padding-top: 5px;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
@dataclass
class SessionState:
    messages: List[Dict[str, str]] = None
    document_processed: bool = False
    vectorstore: Any = None
    qa_mode: bool = False
    model_loaded: bool = False
    model: Any = None
    tokenizer: Any = None
    pipeline: Any = None

    @classmethod
    def initialize(cls):
        if "state" not in st.session_state:
            st.session_state.state = cls(messages=[])


def init_app():
    """Initialize the application state"""
    SessionState.initialize()
    return st.session_state.state


def display_header():
    """Display app header and description"""
    st.title("🤖 Enhanced BlenderBot Conversation Agent")

    with st.expander("About this app", expanded=False):
        st.markdown("""
        This enhanced conversational bot runs locally using:
        - Facebook's BlenderBot 400M distilled model for natural conversations
        - Document Q&A capabilities with PDF, DOCX, TXT, CSV, XLSX support

        **Features:**
        - Natural conversation with context memory
        - Document question-answering with source citation
        - Local processing (no data sent to external APIs)
        """)


@lru_cache(maxsize=1)
def load_model(model_name: str = DEFAULT_MODEL) -> Tuple[Any, Any, Any]:
    """Load the model with caching for efficiency"""
    with st.spinner("Loading BlenderBot model... (this might take a minute on first run)"):
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)
        return model, tokenizer


def create_pipeline(model, tokenizer, temperature, max_length):
    """Create text generation pipeline with current parameters"""
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else device,
        max_length=max_length,
        temperature=temperature
    )


def configure_sidebar(state):
    """Configure the sidebar with model settings and document processing"""
    st.sidebar.header("Model Settings")

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=DEFAULT_TEMP,
        step=0.1,
        help="Higher values = more random responses"
    )

    max_length = st.sidebar.slider(
        "Max Response Length",
        min_value=64,
        max_value=1024,
        value=DEFAULT_MAX_LENGTH,
        step=32,
        help="Maximum number of tokens in response"
    )

    # Update pipeline if parameters changed
    if not state.model_loaded or state.pipeline is None or \
            st.session_state.get('temperature') != temperature or \
            st.session_state.get('max_length') != max_length:

        if not state.model_loaded:
            state.model, state.tokenizer = load_model()
            state.model_loaded = True

        state.pipeline = create_pipeline(
            state.model,
            state.tokenizer,
            temperature,
            max_length
        )
        st.session_state['temperature'] = temperature
        st.session_state['max_length'] = max_length
        st.sidebar.success("✅ Model configured successfully!")

    # Document processing section
    st.sidebar.header("Document Processing")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents for Q&A",
        accept_multiple_files=True,
        type=list(SUPPORTED_FILE_TYPES.keys())
    )

    if uploaded_files and st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            state.vectorstore = process_documents(uploaded_files)
            if state.vectorstore:
                state.document_processed = True
                st.sidebar.success(f"✅ Processed {len(uploaded_files)} documents")
            else:
                st.sidebar.error("❌ Document processing failed")

    # QA mode toggle
    if state.document_processed:
        state.qa_mode = st.sidebar.toggle("Document Q&A Mode", value=state.qa_mode)

        if st.sidebar.button("Clear Documents"):
            state.document_processed = False
            state.vectorstore = None
            state.qa_mode = False
            st.experimental_rerun()

    # Conversation controls
    st.sidebar.header("Conversation Controls")
    if st.sidebar.button("Reset Conversation"):
        state.messages = []
        st.experimental_rerun()

    # Display model information
    st.sidebar.header("Model Information")
    st.sidebar.info(f"""
    **Model Details:**
    - Name: facebook/blenderbot-400M-distill
    - Parameters: 400 million
    - Running on: {device}
    - Temperature: {temperature}
    - Max Length: {max_length} tokens
    """)

    # Display memory usage info
    if state.model_loaded:
        model_size_mb = 400  # Approximate
        st.sidebar.metric("Est. Model Memory", f"{model_size_mb} MB")

        if device.type in ["mps", "cuda"]:
            st.sidebar.success(f"Using hardware acceleration ({device.type.upper()})")


def process_documents(files):
    """Process uploaded documents to create a vector store"""
    if not files:
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        file_progress = st.progress(0)
        total_files = len(files)

        for i, file in enumerate(files):
            file_extension = os.path.splitext(file.name)[1].lower()
            temp_file_path = os.path.join(temp_dir, file.name)

            # Save file
            with open(temp_file_path, "wb") as f:
                f.write(file.getvalue())

            # Get appropriate loader
            loader_class = SUPPORTED_FILE_TYPES.get(file_extension)
            if not loader_class:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            try:
                # Load and process documents
                loader = loader_class(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                st.sidebar.info(f"Loaded {file.name}: {len(docs)} sections")
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
                continue

            # Update progress
            file_progress.progress((i + 1) / total_files)

        if not documents:
            return None

        # Split documents with progress indicator
        with st.spinner("Splitting documents into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            document_chunks = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        with st.spinner("Creating vector embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            vectorstore = FAISS.from_documents(document_chunks, embeddings)

        return vectorstore


def get_qa_chain(state):
    """Create or get the Q&A chain for document interaction"""
    if not state.pipeline or not state.vectorstore:
        return None

    # Create LLM wrapper
    llm = HuggingFacePipeline(pipeline=state.pipeline)

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=state.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return qa_chain


def generate_response(state, prompt):
    """Generate a response based on the current mode (QA or conversation)"""
    if not state.model_loaded or not state.pipeline:
        return "Model is not loaded yet. Please wait or check for errors."

    try:
        if state.qa_mode and state.vectorstore:
            # Document Q&A mode
            qa_chain = get_qa_chain(state)
            if not qa_chain:
                return "Error initializing QA system. Please try again."

            result = qa_chain({"question": prompt})
            response = result["answer"]

            # Add sources if available
            if "source_documents" in result and result["source_documents"]:
                sources_text = "\n\n**Sources:**\n"
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    page_info = f" (page {page})" if page else ""
                    sources_text += f"{i}. {os.path.basename(source)}{page_info}\n"
                response += sources_text
        else:
            # Regular conversation mode
            result = state.pipeline(prompt)
            response = result[0]['generated_text']

        return response

    except Exception as e:
        return f"An error occurred: {str(e)}\n\nThis could be due to model limitations or resource constraints."


def display_chat(state):
    """Display the chat interface and handle interactions"""
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("What would you like to talk about?")

    if prompt:
        # Add user message to chat history
        state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(state, prompt)
                st.markdown(response)

        # Add assistant response to chat history
        state.messages.append({"role": "assistant", "content": response})


def main():
    # Initialize app state
    state = init_app()

    # Display header
    display_header()

    # Configure sidebar
    configure_sidebar(state)

    # Display chat interface
    display_chat(state)


if __name__ == "__main__":
    main()