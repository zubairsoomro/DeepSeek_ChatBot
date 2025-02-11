import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from streamlit_mic_recorder import mic_recorder
import PyPDF2
from io import StringIO
from PIL import Image
import pytesseract
from docx import Document
import pandas as pd
import uuid
import speech_recognition as sr

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Inject custom CSS
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div {
        background-color: #8cc6f4;
        padding: 10px;
    }
    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# UI Setup
st.title("DeepSeek: AI ChatBot")
st.caption("🚀 Your AI Assistant with Debugging and Problem-Solving Powers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:latest"], index=0)
    show_thinking = st.toggle("Show Thinking Messages", value=False)
    tesseract_path = st.text_input("Tesseract Path", value=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    st.markdown("Built with Ollama & LangChain")

llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI assistant. Provide concise and correct answers."
)

# Persistent state for message log
if "message_log" not in st.session_state:
    st.session_state.message_log = []
if not st.session_state.message_log:
    st.session_state.message_log.append({"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you today?"})

# Ensure selected_document_type is initialized
if "selected_document_type" not in st.session_state:
    st.session_state.selected_document_type = "PDF"

chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

voice_query = mic_recorder(start_prompt="🎤 Speak your question", stop_prompt="⏹️ Stop recording", key="recorder")

if voice_query:
    recognizer = sr.Recognizer()
    with sr.AudioFile(voice_query["path"]) as source:
        audio = recognizer.record(source)
    try:
        user_query = recognizer.recognize_google(audio)
        st.session_state.message_log.append({"role": "user", "content": user_query})
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError:
        st.error("Sorry, there was an issue with the speech recognition service.")

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# Document type selection
st.session_state.selected_document_type = st.selectbox(
    "Select Document Type",
    options=["PDF", "Image", "TXT", "DOCX", "CSV"],
    index=["PDF", "Image", "TXT", "DOCX", "CSV"].index(st.session_state.selected_document_type),
    key="document_type_selection"
)

# Set file uploader based on document type
uploaded_file = st.file_uploader(f"Upload a {st.session_state.selected_document_type} file", 
    type=["pdf"] if st.session_state.selected_document_type == "PDF" else 
         ["png", "jpg", "jpeg"] if st.session_state.selected_document_type == "Image" else 
         ["txt"] if st.session_state.selected_document_type == "TXT" else 
         ["docx"] if st.session_state.selected_document_type == "DOCX" else 
         ["csv"]
)

# Process uploaded file
if uploaded_file is not None:
    st.session_state.file_processed = False  # Reset the flag
    extracted_text = ""
    
    if st.session_state.selected_document_type == "PDF":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    elif st.session_state.selected_document_type == "Image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        extracted_text = pytesseract.image_to_string(image)
    
    elif st.session_state.selected_document_type == "TXT":
        extracted_text = uploaded_file.getvalue().decode("utf-8")
    
    elif st.session_state.selected_document_type == "DOCX":
        doc = Document(uploaded_file)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
    
    elif st.session_state.selected_document_type == "CSV":
        df = pd.read_csv(uploaded_file)
        extracted_text = df.to_string()
    
    if extracted_text.strip():
        st.session_state.message_log.append({"role": "ai", "content": f"Extracted text: {extracted_text}"})
        st.session_state.file_processed = True
    else:
        st.warning("No text detected in the uploaded file.")

# Input handling for user queries
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "ai":
            prompt_sequence.append({"role": "assistant", "content": msg["content"]})
    return ChatPromptTemplate.from_messages(prompt_sequence)

user_query = st.chat_input("Type your question here...", key="user_chat_input")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    if show_thinking:
        with st.spinner("🧠 Thinking..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain)
    else:
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
