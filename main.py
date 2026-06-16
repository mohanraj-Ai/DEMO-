print("Starting Application...")

import os

from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

# =====================================
# LOAD ENV
# =====================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print("API KEY FOUND:", bool(GOOGLE_API_KEY))

# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)

# =====================================
# LOAD GEMINI
# =====================================

print("Loading Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

print("Gemini Loaded")

# =====================================
# LOAD EMBEDDINGS
# =====================================

print("Loading Embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

print("Embeddings Loaded")

# =====================================
# LOAD FAISS
# =====================================

print("Loading FAISS...")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS Loaded")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever Ready")

# =====================================
# CHAT MEMORY
# =====================================

chat_history = []

# =====================================
# HOME PAGE
# =====================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        answer="",
        question=""
    )

# =====================================
# ASK ROUTE
# =====================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        question = request.form.get("question")

        print("Question:", question)

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        response = llm.invoke(
            f"""
You are Arrow Solar AI Assistant.

Use only the document context below.

Document Context:
{context}

Question:
{question}

Answer:
"""
        )

        answer = response.content

        return render_template(
            "index.html",
            answer=answer,
            question=question
        )

    except Exception as e:

        print("ERROR:", str(e))

        return render_template(
            "index.html",
            answer=f"Error: {str(e)}",
            question=""
        )

# =====================================
# START SERVER
# =====================================

if __name__ == "__main__":

    print("Starting Flask Server...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )