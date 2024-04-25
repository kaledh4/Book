import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set up the Streamlit app
st.set_page_config(page_title="PDF Chat Agent", layout="wide")
st.title("📚 PDF Chat Agent")

# Add this line to enable Google OAuth
st.session_state.allow_oauth = True

# Create an instance of HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize the FAISS vectorstore
vectorstore = None

# Sidebar for editing the knowledge base
with st.sidebar:
    st.header("📝 Edit Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF Books", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if vectorstore is None:
            # Initialize the FAISS vectorstore if it hasn't been created
            vectorstore = FAISS.from_documents([], embeddings)

        for uploaded_file in uploaded_files:
            # Extract text from the uploaded PDF file
            pdf_reader = PdfReader(uploaded_file)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                texts.append(text)

            # Split the text into chunks
            documents = text_splitter.create_documents(texts)

            # Add the documents to the vectorstore
            vectorstore.add_documents(documents)

        st.success("PDF books have been added to the knowledge base.")

# Chat interface
st.header("💬 Chat with the PDF Books")

# Display chat history
for chat in st.session_state.chat_history[-15:]:
    for qa in chat[-15:]:
        # Display user's question
        st.markdown(f"<div style='text-align: right;'><img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='30'> {qa['question']}</div>", unsafe_allow_html=True)

        # Display AI's answer
        st.markdown(f"<div style='text-align: left;'><img src='https://cdn-icons-png.flaticon.com/512/4712/4712027.png' width='30'> {qa['answer']}</div>", unsafe_allow_html=True)

st.write("---")

# Get user input
question = st.text_input("Enter your question:")
if question:
    if vectorstore is None:
        st.warning("Please upload PDF files to create the knowledge base.")
    else:
        # Find the relevant documents based on the question
        relevant_docs = vectorstore.similarity_search(question)

        # Extract the relevant text from the documents
        relevant_text = " ".join([doc.page_content for doc in relevant_docs])

        # Generate an answer using the question-answering pipeline
        answer = qa_pipeline(question=question, context=relevant_text)

        # Add the question-answer pair to the current chat
        if len(st.session_state.chat_history) == 0 or len(st.session_state.chat_history[-1]) >= 15:
            st.session_state.chat_history.append([])
        st.session_state.chat_history[-1].append({"question": question, "answer": answer["answer"]})

        # Display the answer
        st.markdown(f"<div style='text-align: left;'><img src='https://cdn-icons-png.flaticon.com/512/4712/4712027.png' width='30'> {answer['answer']}</div>", unsafe_allow_html=True)
