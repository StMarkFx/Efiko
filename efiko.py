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
            # Show progress bar for file upload
            progress_bar = st.progress(0)
            chunk_size = 3 * 1024 * 1024  # 3 MB chunks
            total_size = file.size
            bytes_read = 0
            
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                temp_file.write(chunk)
                bytes_read += len(chunk)
                progress_bar.progress(min(bytes_read / total_size, 1.0))
            
            temp_file_path = temp_file.name
            progress_bar.empty()  # Remove progress bar after completion

        # Process different file types
        st.info(f"Processing {file.name}...")
        
        if file.name.lower().endswith('.pdf'):
            texts = process_pdf(temp_file_path)
        elif file.name.lower().endswith('.docx'):
            loader = UnstructuredFileLoader(temp_file_path)
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
        elif file.name.lower().endswith('.txt'):
            with open(temp_file_path, 'r', encoding='utf-8', errors='replace') as f:
                texts = f.readlines()
        else:
            st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
            return None

        # Improved text splitting with semantic units
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize splitting at paragraph and sentence boundaries
        )
        
        combined_text = '\n'.join(texts)
        total_chars = len(combined_text)
        
        # Show progress during text splitting
        with st.spinner("Splitting text into chunks..."):
            split_texts = text_splitter.split_text(combined_text)
        
        st.info(f"Created {len(split_texts)} text chunks for processing")
        
        # Create FAISS index with progress tracking
        vectorstore = None
        batch_size = 500  # Smaller batch size for more frequent updates
        progress_bar = st.progress(0)
        
        for i in range(0, len(split_texts), batch_size):
            batch = split_texts[i:i+batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_texts(batch, embeddings)
            else:
                vectorstore.add_texts(batch)
            
            # Update progress
            progress = min((i + batch_size) / len(split_texts), 1.0)
            progress_bar.progress(progress)
            
        progress_bar.empty()
        
        # Store metadata with the vectorstore
        vectorstore.metadata = {
            "filename": file.name,
            "date_processed": datetime.now().isoformat(),
            "chunk_count": len(split_texts),
            "total_chars": total_chars
        }
        
        return vectorstore
        
    except ImportError as e:
        st.error(f"Error: Missing dependencies for processing this file type. {str(e)}")
        st.info("Please install the required packages by running:")
        st.code("pip install pypdf faiss-cpu unstructured")
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

def get_gemini_response(conversation_buffer, prompt, vectorstore=None, study_profile=None):
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
    """

    # Add user profile information if available
    if study_profile:
        profile_context = f"""
        User Information:
        - Preferred difficulty level: {study_profile.preference_difficulty}
        - Learning style: {study_profile.learning_style}
        - Top studied topics: {', '.join([topic[0] for topic in study_profile.get_top_topics()])}
        - Study time: {study_profile.get_study_stats()['total_time']} minutes
        
        Tailor your response to match the user's preferred difficulty level and learning style.
        For {study_profile.learning_style} learners, emphasize {'visual analogies and diagrams' if study_profile.learning_style == 'visual' 
                                            else 'spoken explanations and discussions' if study_profile.learning_style == 'auditory'
                                            else 'written explanations and examples' if study_profile.learning_style == 'reading'
                                            else 'practical, hands-on examples and activities'}.
        """
        base_context += profile_context
    
    conversation_context = conversation_buffer.get_context()
    
    # Enhanced document search with metadata
    if vectorstore:
        try:
            # First attempt: Direct search on user query
            relevant_docs = vectorstore.similarity_search(prompt, k=3)
            
            # Second attempt: If results are weak, try with expanded query
            if len(relevant_docs) < 2:
                # Extract key terms and create expanded query
                key_terms = extract_topics_from_message(prompt)
                expanded_query = prompt + " " + " ".join(key_terms)
                relevant_docs = vectorstore.similarity_search(expanded_query, k=3)
            
            # Format document context with citations
            doc_context = ""
            for i, doc in enumerate(relevant_docs):
                doc_context += f"\nDocument section {i+1}:\n{doc.page_content}\n"
                
            # Add metadata if available
            if hasattr(vectorstore, 'metadata') and vectorstore.metadata:
                doc_context += f"\nSource: {vectorstore.metadata.get('filename', 'Uploaded document')}"
                
            full_context = f"{base_context}\n{conversation_context}\n\nRelevant document content:\n{doc_context}\n\nUser query: {prompt}"
        except Exception as e:
            st.warning(f"Document search failed: {str(e)}")
            full_context = f"{base_context}\n{conversation_context}\n\nUser query: {prompt}"
    else:
        full_context = f"{base_context}\n{conversation_context}\n\nUser query: {prompt}"

    try:
        # Ensure the model is correctly initialized with safety settings
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Add structured response formatting for certain types of queries
        if any(keyword in prompt.lower() for keyword in ["summarize", "summary", "explain", "explain to me", "break down"]):
            full_context += "\n\nPlease structure your response with clear headings and bullet points when appropriate."
        
        # Add study tips for questions asking how to learn/study
        if any(keyword in prompt.lower() for keyword in ["how to study", "how to learn", "study tips", "study method", "memorize"]):
            full_context += "\n\nInclude practical study tips and memory techniques in your response."
        
        # Generate response with enhanced parameters
        response = model.generate_content(
            full_context,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=800,
            )
        )
        
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
            st.markdown(f"{message['content']}")

    # React to user input
    if prompt := st.chat_input("Ask Efiko about any subject..."):
        st.chat_message("user").markdown(f"{prompt} - {get_current_time()}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_buffer.add_message("user", prompt)

        with st.spinner("Efiko is thinking..."):
            vectorstore = st.session_state.vectorstore
            response = get_gemini_response(st.session_state.conversation_buffer, prompt, vectorstore)

        st.chat_message("assistant").markdown(f"{response} - {get_current_time()}")
        st.session_state.messages.append({"role": "assistant", "content": response})
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


def create_flashcards(text, num_cards=5):
    """Generate flashcards from document text"""
    flashcard_prompt = f"""
    Create {num_cards} educational flashcards from the following text. 
    For each flashcard, generate a question on one side and the answer on the other.
    Format as a JSON array of objects with 'question' and 'answer' fields.
    
    Text: {text[:5000]}  # Limit text length
    
    JSON format: [{"question": "Question 1?", "answer": "Answer 1"}, ...]
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(flashcard_prompt)
        response_text = response.text
        
        # Extract JSON content
        import json
        import re
        
        # Find JSON content (between square brackets)
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            flashcards = json.loads(json_str)
            return flashcards
        else:
            return []
    except Exception as e:
        st.error(f"Error creating flashcards: {str(e)}")
        return []

def generate_quiz(text, num_questions=5):
    """Generate a quiz from document text"""
    quiz_prompt = f"""
    Create a {num_questions}-question multiple-choice quiz based on the following text.
    For each question, provide 4 options with one correct answer clearly marked.
    Format as a JSON array of objects with 'question', 'options' (array), and 'correct_index' fields.
    
    Text: {text[:5000]}  # Limit text length
    
    JSON format example: 
    [
      {{
        "question": "What is the capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "correct_index": 2
      }}
    ]
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(quiz_prompt)
        response_text = response.text
        
        # Extract JSON content
        import json
        import re
        
        # Find JSON content (between square brackets)
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            quiz = json.loads(json_str)
            return quiz
        else:
            return []
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []

def summarize_document(text):
    """Generate a summary of the document"""
    summary_prompt = f"""
    Create a comprehensive summary of the following text. 
    Include the main topics, key points, and important concepts.
    Organize the summary with clear headings and bullet points.
    
    Text: {text[:7000]}  # Limit text length
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Could not generate summary."


#Study Tools Features
def study_tools_tab():
    """Create a study tools interface tab"""
    st.header("Study Tools")
    
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.info("Please upload a document first to use study tools.")
        return
    
    # Get document text
    try:
        vectorstore = st.session_state.vectorstore
        # Get a sample of document chunks for tools
        sample_docs = vectorstore.similarity_search("important concepts", k=10)
        doc_text = "\n\n".join([doc.page_content for doc in sample_docs])
        
        # Tool selection
        tool = st.selectbox(
            "Select Study Tool",
            ["Document Summary", "Flashcards", "Practice Quiz"]
        )
        
        if tool == "Document Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating document summary..."):
                    summary = summarize_document(doc_text)
                    st.markdown(summary)
                    
                    # Add download option
                    from io import StringIO
                    summary_buffer = StringIO()
                    summary_buffer.write(summary)
                    st.download_button(
                        label="Download Summary",
                        data=summary_buffer.getvalue(),
                        file_name="document_summary.txt",
                        mime="text/plain"
                    )
        
        elif tool == "Flashcards":
            num_cards = st.slider("Number of Flashcards", 5, 20, 10)
            if st.button("Generate Flashcards"):
                with st.spinner("Creating flashcards..."):
                    flashcards = create_flashcards(doc_text, num_cards)
                    
                    if flashcards:
                        # Display flashcards in a carousel-like interface
                        selected_card = st.session_state.get("selected_card", 0)
                        col1, col2, col3 = st.columns([1, 10, 1])
                        
                        with col1:
                            if st.button("◀️", key="prev_card") and selected_card > 0:
                                st.session_state.selected_card = selected_card - 1
                                st.experimental_rerun()
                        
                        with col3:
                            if st.button("▶️", key="next_card") and selected_card < len(flashcards) - 1:
                                st.session_state.selected_card = selected_card + 1
                                st.experimental_rerun()
                        
                        # Display current flashcard
                        current_card = flashcards[st.session_state.get("selected_card", 0)]
                        with col2:
                            with st.container():
                                st.subheader(f"Flashcard {selected_card + 1}/{len(flashcards)}")
                                show_answer = st.checkbox("Show Answer", key=f"show_answer_{selected_card}")
                                st.info(current_card["question"])
                                if show_answer:
                                    st.success(current_card["answer"])
                    else:
                        st.warning("Could not generate flashcards. Please try again.")
        
        elif tool == "Practice Quiz":
            num_questions = st.slider("Number of Questions", 3, 10, 5)
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz..."):
                    quiz = generate_quiz(doc_text, num_questions)
                    
                    if quiz:
                        # Initialize session state for quiz
                        if "quiz_answers" not in st.session_state:
                            st.session_state.quiz_answers = [-1] * len(quiz)
                            st.session_state.show_results = False
                        
                        # Display quiz
                        for i, question in enumerate(quiz):
                            st.subheader(f"Question {i+1}")
                            st.write(question["question"])
                            
                            # Radio buttons for options
                            selected = st.radio(
                                "Select your answer:",
                                question["options"],
                                key=f"q_{i}"
                            )
                            
                            # Store selected answer index
                            if selected:
                                st.session_state.quiz_answers[i] = question["options"].index(selected)
                            
                            st.markdown("---")
                        
                        # Submit button
                        if st.button("Submit Quiz"):
                            st.session_state.show_results = True
                        
                        # Show results
                        if st.session_state.show_results:
                            score = 0
                            for i, question in enumerate(quiz):
                                user_answer = st.session_state.quiz_answers[i]
                                correct = question["correct_index"]
                                
                                if user_answer == correct:
                                    score += 1
                                    st.success(f"Question {i+1}: Correct! ✓")
                                else:
                                    st.error(f"Question {i+1}: Incorrect ✗")
                                    st.info(f"Correct answer: {question['options'][correct]}")
                            
                            # Display final score
                            percentage = (score / len(quiz)) * 100
                            st.subheader(f"Your Score: {score}/{len(quiz)} ({percentage:.1f}%)")
                            
                            # Reset button
                            if st.button("Try Again"):
                                st.session_state.quiz_answers = [-1] * len(quiz)
                                st.session_state.show_results = False
                                st.experimental_rerun()
                    else:
                        st.warning("Could not generate quiz. Please try again.")
                        
    except Exception as e:
        st.error(f"Error loading document content: {str(e)}")



if __name__ == "__main__":
    chat_interface()
