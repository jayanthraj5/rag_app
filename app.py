import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("PDF RAG App")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

    return embeddings, llm

embeddings, llm = load_models()

pdf_path = "sample.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

vector_db = FAISS.from_documents(chunks, embeddings)

query = st.text_input("Ask a question")

if query:
    results = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
Use only the context below to answer.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm(prompt)
    st.write(response[0]["generated_text"])