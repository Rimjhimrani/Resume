import streamlit as st
import os
import re
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# --- Helper Functions from Original Script ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=256):
    """Splits text into smaller, overlapping chunks based on sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text) # Split by sentences
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk]

def calculate_match_score(resume_text, jd_text):
    """Calculates a match score using TF-IDF and Cosine Similarity."""
    if not resume_text or not jd_text:
        return 0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return int(cosine_sim[0][0] * 100)

def extract_strengths_gaps(resume_text, jd_text):
    """Uses LLM to find strengths and gaps."""
    prompt = f"""
    Analyze the following resume and job description.
    Based ONLY on the information provided in the resume against the job description, identify the candidate's key strengths and gaps.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Format your response as:
    Strengths:
    - [Strength 1]
    - [Strength 2]
    ...
    Gaps:
    - [Gap 1]
    - [Gap 2]
    ...
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful recruitment assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with OpenAI: {e}")
        return "Could not retrieve insights."

class RAGSystem:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index, self.chunks = self._build_index()

    def _build_index(self):
        """Creates a FAISS index for the document chunks."""
        all_chunks = []
        for doc in self.documents:
            all_chunks.extend(chunk_text(doc))
        
        if not all_chunks:
            return None, []

        embeddings = self.model.encode(all_chunks, convert_to_tensor=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index, all_chunks

    def query(self, question, k=3):
        """Retrieves relevant context for a given question."""
        if self.index is None:
            return "No documents to search."
            
        question_embedding = self.model.encode([question])
        distances, indices = self.index.search(question_embedding, k)
        return " ".join([self.chunks[i] for i in indices[0]])

    def ask_question(self, question):
        """Performs the full RAG process: retrieve then generate."""
        retrieved_context = self.query(question)
        
        prompt = f"""
        Based STRICTLY on the following context from the resume, please answer the user's question.
        If the context does not contain the answer, state that the information is not found in the resume.

        Context:
        {retrieved_context}

        Question:
        {question}

        Answer:
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant answering questions about a resume based only on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI: {e}"


# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📄 AI-Powered Resume Screening Tool")

# Initialize session state variables
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None


with st.sidebar:
    st.header("Setup")
    # Using st.secrets for production or environment variables is better.
    # For this demo, a text input is used.
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

    st.header("Upload Files")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])
    jd_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"])

    if st.button("Analyze"):
        if not api_key:
            st.warning("Please enter your OpenAI API Key.")
        elif resume_file and jd_file:
            with st.spinner("Processing documents..."):
                openai.api_key = api_key
                # Extract text
                if resume_file.type == "application/pdf":
                    st.session_state.resume_text = extract_text_from_pdf(resume_file)
                else:
                    st.session_state.resume_text = resume_file.read().decode("utf-8")

                if jd_file.type == "application/pdf":
                    st.session_state.jd_text = extract_text_from_pdf(jd_file)
                else:
                    st.session_state.jd_text = jd_file.read().decode("utf-8")

                # Initialize RAG system and store in session state
                st.session_state.rag_system = RAGSystem([st.session_state.resume_text])
                
                # Mark analysis as done
                st.session_state.analysis_done = True
                # Clear previous chat
                st.session_state.messages = [] 
                st.success("Analysis complete!")
        else:
            st.warning("Please upload both a resume and a job description.")

if st.session_state.analysis_done:
    # --- Main Analysis Display ---
    st.header("Match Analysis")
    
    # Calculate score and insights
    match_score = calculate_match_score(st.session_state.resume_text, st.session_state.jd_text)
    insights = extract_strengths_gaps(st.session_state.resume_text, st.session_state.jd_text)
    
    # Parse insights for display
    try:
        strengths = insights.split("Strengths:")[1].split("Gaps:")[0]
        gaps = insights.split("Gaps:")[1]
    except IndexError:
        strengths = "Could not parse strengths."
        gaps = "Could not parse gaps."

    st.metric(label="**Match Score**", value=f"{match_score}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Strengths")
        st.markdown(strengths)
    with col2:
        st.subheader("❌ Gaps")
        st.markdown(gaps)

    st.divider()

    # --- Chat Interface ---
    st.header("Ask Questions About This Candidate")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the resume..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display AI response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_system = st.session_state.rag_system
                response = rag_system.ask_question(prompt)
                st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Upload a resume and job description and click 'Analyze' to begin.")
