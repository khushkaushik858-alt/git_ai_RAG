from openai import OpenAI
from rag_engine import create_vector_db, load_vector_db
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# First time → build DB
if not os.path.exists("vector_db"):
    create_vector_db()

db = load_vector_db()
retriever = db.as_retriever(search_kwargs={"k": 4})


def ask_question(q):
    docs = retriever.invoke(q)
    context = "\n\n".join([d.page_content for d in docs])

    response = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a CBSE Class 10 English Board Examiner. Write answers in board exam format using simple formal language."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {q}\n\nGive exam ready answer."
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


print("\n📘 CBSE English AI Ready! Type exit to quit")

while True:
    q = input("\nAsk Question: ")

    if q.lower() == "exit":
        break

    ans = ask_question(q)
    print("\n📝 Answer:\n", ans)