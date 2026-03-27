import os
from dotenv import load_dotenv

#  NEW GEMINI SDK
from google import genai

#  NEW LANGCHAIN IMPORTS (2026 STRUCTURE)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DATA_PATH = "CBSE_Class10_English"

# LOAD ALL PDF FILES
def load_all_pdfs(base_folder):
    docs = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                print("Loading:", path)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
    return docs

#  SPLIT TEXT INTO CHUNKS
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

#  CREATE VECTOR DATABASE (FREE EMBEDDINGS)
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("cbse_index")
    print(" Vector DB created!")

#  LOAD EXISTING DATABASE
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local("cbse_index", embeddings, allow_dangerous_deserialization=True)

#  GEMINI ANSWER GENERATOR
def generate_answer(question, context):
    prompt = f"""
You are a CBSE Class 10 English Board Examiner.

STRICT RULES:
• Answer like a board exam answer sheet
• Keep language simple and formal
• Stay concise and relevant
• Use textbook wording when possible

Context:
{context}

Student Question:
{question}

Give the best board exam answer.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

# FULL QUESTION PIPELINE
def ask_question(question):
    db = load_vector_db()
    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    return generate_answer(question, context)

# BUILD DATABASE FIRST TIME
def build_database():
    print("\n📘 Loading CBSE PDFs...")
    docs = load_all_pdfs(DATA_PATH)

    print("✂️ Splitting text...")
    chunks = split_documents(docs)

    print("🧠 Creating embeddings (1–2 mins)...")
    create_vector_db(chunks)

    print("\n🎉 DATABASE READY!")

#  MAIN PROGRAM
if __name__ == "__main__":

    if not os.path.exists("cbse_index"):
        build_database()

    print("\n📘 CBSE English AI Ready! Type 'exit' to quit.\n")

    while True:
        question = input("Ask Question: ")
        if question.lower() == "exit":
            break

        answer = ask_question(question)
        print("\n📝 Answer:\n", answer, "\n")