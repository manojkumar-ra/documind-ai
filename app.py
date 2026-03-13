import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set API key
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

# App title
st.title("DocuMind AI")
st.subheader("Chat with your documents using AI")

# File upload
uploaded_file = st.file_uploader("Upload your document", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.create_documents([text])
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings)
    
    st.success("Document loaded successfully!")
    
    question = st.text_input("Ask a question about your document:")
    
    if question:
        with st.spinner("Thinking..."):
            docs = vectordb.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            llm = ChatGroq(model="llama-3.3-70b-versatile")
            answer = llm.invoke(f"Answer this question: {question}\n\nContext: {context}")
            st.write("Answer:", answer.content)