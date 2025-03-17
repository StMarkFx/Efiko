import streamlit as st
from datetime import datetime
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from dotenv import load_dotenv
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT
from io import BytesIO
from PIL import Image
import pickle
from PyPDF2 import PdfReader

load_dotenv()

try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
except ImportError:
    st.error("Error: Some required packages are missing. Please install the required packages.")
    st.info("Run the following command to install the necessary packages:")
    st.code("pip install langchain chromadb sentence_transformers")
    st.stop()

st.set_page_config(
    page_title="Efiko - Your Study Buddy",
    page_icon="efiko.jpg",  
    layout="centered",  
    initial_sidebar_state="expanded"
)

# Access the Gemini API key
gemini_api_key = st.secrets["gemini"]["api_key"]

#st.write(f"API Key from secrets: {st.secrets['gemini']['api_key'][:5]}...")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

def get_current_time():
    return datetime.now().strftime("%H:%M")

def process_document(file):
    MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit
    if file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the limit of 15MB. Please upload a smaller file.")
        return None

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            # Read and write the file in chunks
            chunk_size = 3 *1024 * 1024  # 3 MB chunks
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        if file.name.endswith('.pdf'):
            texts = process_pdf(temp_file_path)
        elif file.name.endswith('.docx'):
            loader = UnstructuredFileLoader(temp_file_path)
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
        elif file.name.endswith('.txt'):
            with open(temp_file_path, 'r') as f:
                texts = f.readlines()
        else:
            st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_texts = text_splitter.split_text('\n'.join(texts))
        
        # Create FAISS index in batches
        vectorstore = None
        batch_size = 1000
        for i in range(0, len(split_texts), batch_size):
            batch = split_texts[i:i+batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_texts(batch, embeddings)
            else:
                vectorstore.add_texts(batch)
        
        return vectorstore
    except ImportError as e:
        st.error("Error: Missing dependencies for processing this file type.")
        st.info("Please install the required packages by running:")
        st.code("pip install pypdf faiss-cpu")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")

def process_pdf(file_path):
    texts = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            texts.append(page.extract_text())
    return texts

def save_vectorstore(vectorstore, filename="vectorstore.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(filename="vectorstore.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

class ConversationBuffer:
    def __init__(self, max_turns=5):
        self.buffer = []
        self.max_turns = max_turns

    def add_message(self, role, content):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > self.max_turns * 2:  # *2 because each turn has a user and an assistant message
            self.buffer = self.buffer[-self.max_turns * 2:]

    def get_context(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.buffer])

def safe_get_gemini_response(conversation_buffer, prompt, vectorstore=None):
    try:
        return get_gemini_response(conversation_buffer, prompt, vectorstore)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please try again.")
        return "I'm sorry, I encountered an error. Could you please rephrase your question?"

def get_gemini_response(conversation_buffer, prompt, vectorstore=None):
    base_context = """You are Efiko, an enthusiastic and knowledgeable AI study companion. Your goal is to inspire curiosity and a love for learning in students of all ages and backgrounds. When interacting with users:

    1. Be proactive and engaging. If a query is vague, offer a range of exciting topics or suggest a learning path based on current events or interdisciplinary connections.
    2. Adapt your language and explanations to suit different age groups and learning levels.
    3. Provide concise but informative responses, always aiming to spark further interest in the topic.
    4. Offer practical examples and real-world applications of concepts to make learning relatable.
    5. Encourage critical thinking by posing thought-provoking questions related to the topic.
    6. If appropriate, suggest fun learning activities or experiments that can be done at home.
    7. Be supportive and motivational, acknowledging the user's interest in learning.
    8. If you don't have specific information, guide the user towards reliable resources or suggest how they might research the topic further.

    Remember, your role is not just to provide information, but to inspire a journey of discovery and lifelong learning.

    Previous conversation:
    """
    conversation_context = conversation_buffer.get_context()
    
    if vectorstore:
        relevant_docs = vectorstore.similarity_search(prompt, k=2)
        doc_context = "\n".join([doc.page_content for doc in relevant_docs])
        full_context = f"{base_context}\n{conversation_context}\n\nRelevant document content:\n{doc_context}\n\nUser query: {prompt}"
    else:
        full_context = f"{base_context}\n{conversation_context}\n\nUser query: {prompt}"

    try:
        # Ensure the model is correctly initialized
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(full_context)
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return ' '.join(part.text for part in response.parts)
        else:
            return str(response)
    except Exception as e:
        st.error(f"An error occurred while generating the response: {str(e)}")
        return "I'm sorry, I encountered an error. Could you please try again or rephrase your question?"


def export_conversation_to_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Create custom styles for "You" and "Efiko"
    styles.add(ParagraphStyle(name='You', parent=styles['Normal'], spaceAfter=10, textColor='blue'))
    styles.add(ParagraphStyle(name='Efiko', parent=styles['Normal'], spaceAfter=10, textColor='green', alignment=TA_RIGHT))

    for message in st.session_state.messages:
        if message['role'] == 'user':
            story.append(Paragraph(f"You: {message['content']}", styles['You']))
        else:
            story.append(Paragraph(f"Efiko: {message['content']}", styles['Efiko']))
        story.append(Spacer(1, 12))

    doc.build(story)
    return buffer

def chat_interface():
    st.title("Efiko - Your Study Companion")
    st.subheader("Ask me anything about your studies!")

    # Initialize vectorstore in session state if it doesn't exist
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()

    # Sidebar for logo, document upload and session management
    with st.sidebar:
        # Add the logo at the top of the sidebar
        logo = Image.open("efiko.jpg")
        st.image(logo, width=150)  # Adjust width as needed

        st.markdown("""
        ### Hi, I'm Efiko, your smart study companion! 
        I'm here to help you learn and understand various subjects. Ask me anything!

        ---

        Efiko is a study chatbot built by St. Mark Adebayo. 

        ---

        St. Mark Adebayo is a Data Science Fellow at 3MTT Nigeria. He is also a Machine Learning Engineer and AI Enthusiast. This project is his submission for the September Knowledge Showcase.

        ---
        """)

        st.header("Document Upload")
        MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'], 
                                         accept_multiple_files=False,
                                         help=f"Max file size: 15MB")
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the limit of 15MB. Please upload a smaller file.")
            else:
                # Check if the file has changed
                if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                    with st.spinner("Processing document..."):
                        vectorstore = process_document(uploaded_file)
                    if vectorstore is not None:
                        st.session_state.vectorstore = vectorstore
                        save_vectorstore(vectorstore)
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.success("Document processed and saved successfully!")
                    else:
                        st.warning("Document processing failed. Please check the error message above.")
                else:
                    st.info("Document already processed. Using existing vectorstore.")

        st.header("Session Management")
        
        if st.button("Save Chat Session"):
            st.session_state.saved_session = {
                "messages": st.session_state.messages,
                "conversation_buffer": st.session_state.conversation_buffer
            }
            st.success("Session saved successfully!")
        
        if st.button("Load Last Session"):
            if "saved_session" in st.session_state:
                st.session_state.messages = st.session_state.saved_session["messages"]
                st.session_state.conversation_buffer = st.session_state.saved_session["conversation_buffer"]
                st.success("Session loaded successfully!")
            else:
                st.warning("No saved session found.")
        
        if st.button("Export Conversation"):
            pdf_buffer = export_conversation_to_pdf()
            st.download_button(
                label="Download Conversation as PDF",
                data=pdf_buffer.getvalue(),
                file_name="conversation.pdf",
                mime="application/pdf"
            )

    # Chat area
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_buffer" not in st.session_state:
        st.session_state.conversation_buffer = ConversationBuffer()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']} - {message['time']}")

    # React to user input
    if prompt := st.chat_input("Ask Efiko about any subject..."):
        st.chat_message("user").markdown(f"{prompt} - {get_current_time()}")
        st.session_state.messages.append({"role": "user", "content": prompt, "time": get_current_time()})
        st.session_state.conversation_buffer.add_message("user", prompt)

        with st.spinner("Efiko is thinking..."):
            vectorstore = st.session_state.vectorstore
            response = get_gemini_response(st.session_state.conversation_buffer, prompt, vectorstore)

        st.chat_message("assistant").markdown(f"{response} - {get_current_time()}")
        st.session_state.messages.append({"role": "assistant", "content": response, "time": get_current_time()})
        st.session_state.conversation_buffer.add_message("assistant", response)

def cleanup_old_vectorstores(max_age_days=2):
    current_time = datetime.now()
    for filename in os.listdir():
        if filename.startswith("vectorstore_") and filename.endswith(".pkl"):
            file_path = os.path.join(os.getcwd(), filename)
            file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
            if file_age.days > max_age_days:
                os.remove(file_path)

# Call this function periodically, e.g., once a day or once a week

if __name__ == "__main__":
    chat_interface()
