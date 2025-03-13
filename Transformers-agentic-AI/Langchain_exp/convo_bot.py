# app.py
import streamlit as st
import requests
import os
import io
import base64
import json
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from gtts import gTTS
from speech_recognition import Recognizer, AudioFile
import pycountry

# Load environment variables
load_dotenv()

# Get API key from environment variables
HUGGING_FACE_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Set the API key for Hugging Face
if HUGGING_FACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY

# Page configuration
st.set_page_config(page_title="Advanced AI Conversation Agent", page_icon="🤖", layout="wide")

# App title and description
st.title("🤖 Advanced AI Conversation Agent")
st.markdown("""
This enhanced conversational bot features:
- Document Q&A capabilities
- Multi-language support
- Speech-to-text and text-to-speech
- External API connections for real-time information
- Long-term memory
""")

# Sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection with appropriate conversation models
model_name = st.sidebar.selectbox(
    "Select Conversation Model",
    [
        "facebook/bart-large",
        "facebook/blenderbot-400M-distill",
        "gpt2",
        "google/flan-t5-large",
        "tiiuae/falcon-7b-instruct"
    ],
    index=0
)

embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ],
    index=0
)

# Adjust max length based on model
if "t5" in model_name:
    default_max_length = 512
    max_max_length = 1024
elif "gpt2" in model_name:
    default_max_length = 256
    max_max_length = 1024
elif "blenderbot" in model_name:
    default_max_length = 128
    max_max_length = 512
elif "falcon" in model_name:
    default_max_length = 512
    max_max_length = 2048
else:
    default_max_length = 256
    max_max_length = 1024

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
max_length = st.sidebar.slider("Max Response Length (tokens)", min_value=64, max_value=max_max_length,
                               value=default_max_length, step=32)

# Language settings
languages = {lang.name: lang.alpha_2 for lang in pycountry.languages if hasattr(lang, 'alpha_2')}
language_options = list(languages.keys())
language_options.sort()
selected_language = st.sidebar.selectbox("Interface Language", language_options,
                                         index=language_options.index("English"))
language_code = languages[selected_language]


# Get translation function
def translate_text(text, target_language):
    # No actual translation happening here - in a production app,
    # you would integrate with a translation API
    if target_language == "en":
        return text

    try:
        # This is a placeholder. In a real app, you would call a translation API
        # For example: response = requests.post("translation-api-url", json={"text": text, "target": target_language})
        # For now, we'll just append a notice about translation
        if target_language != "en":
            return f"{text} [Translation to {target_language} would happen here]"
        return text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


# Speech-to-text and text-to-speech options
enable_voice = st.sidebar.checkbox("Enable Voice Features", value=False)

# API key status indicator
if not HUGGING_FACE_API_KEY:
    st.sidebar.error("Hugging Face API key not found! Please add it to your .env file.")
else:
    st.sidebar.success("Hugging Face API key loaded")

if not WEATHER_API_KEY:
    st.sidebar.warning("Weather API key not found. Weather features will be limited.")

if not NEWS_API_KEY:
    st.sidebar.warning("News API key not found. News features will be limited.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_mode" not in st.session_state:
    st.session_state.qa_mode = False

# File uploader for document processing
st.sidebar.header("Document Processing")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents for Q&A",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "csv", "xlsx"]
)


# Function to process uploaded documents
def process_documents(files):
    if not files:
        return None

    # Create a temporary directory to store the files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each file
        documents = []
        for file in files:
            # Get file extension
            file_extension = os.path.splitext(file.name)[1].lower()

            # Save the file to the temporary directory
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getvalue())

            # Load the document based on file type
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            elif file_extension in [".csv", ".xlsx"]:
                loader = UnstructuredExcelLoader(temp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            # Load documents
            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
                continue

        if not documents:
            return None

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(documents)

        # Create embeddings and vectorstore
        embeddings = HuggingFaceHubEmbeddings(
            repo_id=embedding_model,
            task="feature-extraction"
        )

        vectorstore = FAISS.from_documents(document_chunks, embeddings)
        return vectorstore


# Process documents button
if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = process_documents(uploaded_files)
        if st.session_state.vectorstore:
            st.session_state.document_processed = True
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents")
        else:
            st.sidebar.error("Failed to process documents")

# Toggle for Q&A mode
if st.session_state.document_processed:
    st.session_state.qa_mode = st.sidebar.checkbox("Enable Document Q&A Mode", value=False)

    if st.sidebar.button("Clear Documents"):
        st.session_state.document_processed = False
        st.session_state.vectorstore = None
        st.session_state.qa_mode = False
        st.experimental_rerun()

# External API section
st.sidebar.header("External API Features")
enable_weather = st.sidebar.checkbox("Enable Weather Queries", value=True)
enable_news = st.sidebar.checkbox("Enable News Queries", value=True)


# Initialize LangChain components
@st.cache_resource
def get_conversation_chain():
    # Create a LLM from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={
            "temperature": temperature,
            "max_length": max_length,
        }
    )

    # Create a template for the conversation
    template = """The following is a friendly conversation between a human and an AI assistant.
    The assistant is helpful, creative, and tailors its responses to the human's requests.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
    """

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


def get_qa_chain():
    # Create a LLM from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={
            "temperature": temperature,
            "max_length": max_length,
        }
    )

    # Create a conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return qa_chain


# Functions for external API calls
def get_weather(location):
    """Get current weather for a location using OpenWeatherMap API"""
    if not WEATHER_API_KEY:
        return "Weather API key not configured. Please add it to your .env file to enable this feature."

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            description = data["weather"][0]["description"]
            wind_speed = data["wind"]["speed"]

            weather_info = f"Weather in {location}:\n"
            weather_info += f"Temperature: {temp}°C\n"
            weather_info += f"Humidity: {humidity}%\n"
            weather_info += f"Conditions: {description}\n"
            weather_info += f"Wind Speed: {wind_speed} m/s"

            return weather_info
        else:
            return f"Error getting weather: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error accessing weather service: {str(e)}"


def get_news(topic, count=3):
    """Get latest news on a topic using NewsAPI"""
    if not NEWS_API_KEY:
        return "News API key not configured. Please add it to your .env file to enable this feature."

    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&pageSize={count}"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and data.get("articles"):
            articles = data["articles"][:count]
            news_info = f"Latest news about {topic}:\n\n"

            for i, article in enumerate(articles, 1):
                news_info += f"{i}. {article['title']}\n"
                news_info += f"   Source: {article['source']['name']}\n"
                news_info += f"   Published: {article['publishedAt'][:10]}\n"
                news_info += f"   {article['description']}\n\n"

            return news_info
        else:
            return f"Error getting news: {data.get('message', 'No articles found')}"
    except Exception as e:
        return f"Error accessing news service: {str(e)}"


# Speech-to-text function
def speech_to_text(audio_bytes):
    try:
        # Save the audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Use speech recognition on the audio file
        recognizer = Recognizer()
        with AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Clean up the temporary file
        os.unlink(temp_audio_path)

        return text
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None


# Text-to-speech function
def text_to_speech(text, lang_code='en'):
    try:
        tts = gTTS(text=text, lang=lang_code[:2], slow=False)

        # Save to a BytesIO object
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # Encode to base64 for HTML audio playback
        audio_bytes = fp.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        # Create HTML for the audio player
        audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """

        return audio_html
    except Exception as e:
        st.error(f"Text to speech error: {str(e)}")
        return None


# Function to extract API queries
def extract_api_query(message):
    # Simple keyword-based extraction
    message_lower = message.lower()

    # Check for weather queries
    if enable_weather and any(
            keyword in message_lower for keyword in ["weather in", "temperature in", "what's the weather"]):
        # Extract location - this is a simple implementation
        for keyword in ["weather in", "temperature in"]:
            if keyword in message_lower:
                location = message_lower.split(keyword)[1].strip().split()[0]
                return "weather", location

    # Check for news queries
    if enable_news and any(
            keyword in message_lower for keyword in ["news about", "latest on", "updates on", "articles about"]):
        # Extract topic - this is a simple implementation
        for keyword in ["news about", "latest on", "updates on", "articles about"]:
            if keyword in message_lower:
                topic = message_lower.split(keyword)[1].strip()
                return "news", topic

    return None, None


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Audio input
if enable_voice:
    st.subheader("Voice Input")
    audio_input = st.audio_recorder(text="Click to record")

    if audio_input:
        with st.spinner("Processing audio..."):
            transcribed_text = speech_to_text(audio_input)
            if transcribed_text:
                st.success(f"Transcribed: {transcribed_text}")
                # Use the transcribed text as the prompt
                prompt = transcribed_text
            else:
                st.error("Could not transcribe audio. Please try again or type your message.")
                prompt = None
    else:
        prompt = st.chat_input("What would you like to talk about?")
else:
    prompt = st.chat_input("What would you like to talk about?")

# Chat processing
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if API key is configured
    if not HUGGING_FACE_API_KEY:
        with st.chat_message("assistant"):
            st.error("Please set your Hugging Face API Key in the .env file before running the app.")
    else:
        try:
            # Check for API queries first
            api_type, query_param = extract_api_query(prompt)

            if api_type == "weather" and query_param:
                # Handle weather query
                with st.chat_message("assistant"):
                    with st.spinner(f"Getting weather for {query_param}..."):
                        weather_info = get_weather(query_param)
                        st.markdown(weather_info)

                        # Text-to-speech for the response if enabled
                        if enable_voice:
                            audio_html = text_to_speech(weather_info, language_code)
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": weather_info})

            elif api_type == "news" and query_param:
                # Handle news query
                with st.chat_message("assistant"):
                    with st.spinner(f"Getting news about {query_param}..."):
                        news_info = get_news(query_param)
                        st.markdown(news_info)

                        # Text-to-speech for the response if enabled
                        if enable_voice:
                            audio_html = text_to_speech(news_info, language_code)
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": news_info})

            else:
                # Handle regular conversation or Q&A
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if st.session_state.qa_mode and st.session_state.vectorstore:
                            # Q&A mode with documents
                            qa_chain = get_qa_chain()
                            result = qa_chain({"question": prompt})
                            response = result["answer"]

                            # Display sources if available
                            if "source_documents" in result and result["source_documents"]:
                                response += "\n\nSources:\n"
                                for i, doc in enumerate(result["source_documents"][:3], 1):
                                    source = doc.metadata.get("source", "Unknown")
                                    page = doc.metadata.get("page", "")
                                    page_info = f" (page {page})" if page else ""
                                    response += f"{i}. {os.path.basename(source)}{page_info}\n"
                        else:
                            # Regular conversation mode
                            conversation = get_conversation_chain()
                            response = conversation.predict(input=prompt)

                        # Translate response if needed
                        if language_code != "en":
                            response = translate_text(response, language_code)

                        st.markdown(response)

                        # Text-to-speech for the response if enabled
                        if enable_voice:
                            audio_html = text_to_speech(response, language_code)
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            with st.chat_message("assistant"):
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.info("This could be due to model limitations, API rate limits, or connectivity problems.")

            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add controls in the sidebar
st.sidebar.header("Conversation Controls")
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display token usage information
st.sidebar.header("Token Information")
st.sidebar.info(f"""
**Model Token Capacity:**
- Max Output Tokens: {max_length}
- Approximate max input tokens varies by model:
  - BART: ~1024 tokens
  - GPT-2: ~1024 tokens
  - T5: ~512 tokens
  - Falcon: ~2048 tokens

Note: 1 token ≈ 4 characters or 0.75 words
""")

# Add a footer
st.markdown("""
---
🤖 Advanced AI Conversation Agent | Built with Streamlit, LangChain & Hugging Face
""")

# CSS to improve the appearance
st.markdown("""
<style>
.stChatMessage {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.stSpinner {
    text-align: center;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)