import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "CBSE_Class10_English"
DB_PATH = "vector_db"


# 📚 LOAD ALL PDFs FROM FOLDER (RECURSIVE)
def load_all_pdfs(base_folder):
    documents = []

    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                print("Loading:", path)

                loader = PyPDFLoader(path)
                documents.extend(loader.load())

    return documents


# ✂️ SPLIT INTO CHUNKS
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# 🧠 CREATE VECTOR DATABASE
def create_vector_db():
    print("\n📚 Loading all CBSE English PDFs...")

    docs = load_all_pdfs(DATA_PATH)

    print("✂️ Splitting into chunks...")
    chunks = split_docs(docs)

    print("🧠 Creating embeddings (first run takes 2–5 mins)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    print("\n✅ VECTOR DATABASE READY!")


# 📦 LOAD EXISTING DATABASE
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)