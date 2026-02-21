"""
main.py
--------
FastAPI backend for a RAG-based role-based access chatbot at FinSolve Technologies.

Now uses:
- Google AI Studio (Gemini) for LLM responses
- Google embeddings (text-embedding-004) for semantic search
"""

from typing import Dict
import os

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.vectorstores import Chroma

from app.google_embeddings import GoogleAIStudioEmbeddings

# -----------------------------
# Env + Google Client Setup
# -----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment (.env).")

genai.configure(api_key=GOOGLE_API_KEY)

# Choose Gemini model (fast + good):
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI()
security = HTTPBasic()

# -----------------------------
# Load Vector Database
# -----------------------------
embedding_function = GoogleAIStudioEmbeddings(model="text-embedding-004")

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function,
    collection_name="company_docs",
)

# -----------------------------
# Dummy Users Database
# -----------------------------
users_db: Dict[str, Dict[str, str]] = {
    "Aakanksha": {"password": "password123", "role": "engineering"},
    "Sahithi": {"password": "securepass", "role": "marketing"},
    "Yasasvi": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Shiva": {"password": "hrpass123", "role": "hr"},
    "Gowtham": {"password": "ceopass", "role": "c-levelexecutives"},
    "Kiran": {"password": "employeepass", "role": "employee"},
}

# -----------------------------
# Helper: Authentication
# -----------------------------
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    # DEBUG prints (requested)
    print("DEBUG username:", repr(credentials.username))
    print("DEBUG known users:", list(users_db.keys()))

    username = credentials.username
    password = credentials.password

    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"username": username, "role": user["role"]}

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint:
    - Retrieves relevant documents from ChromaDB based on user role
    - Uses Google embeddings for semantic search
    - Uses Gemini to generate the final answer
    """
    try:
        data = await request.json()
        user = data["user"]
        message = data["message"]
        user_role = user["role"].lower()

        # -----------------------------
        # Role-based retrieval
        # -----------------------------
        if "c-levelexecutives" in user_role:
            docs = vectordb.similarity_search(message, k=3)
            if not docs:
                docs = vectordb.similarity_search(
                    message,
                    k=5,
                    filter={"role": {"$in": ["engineering", "hr", "finance", "marketing", "general"]}},
                )

        elif "employee" in user_role:
            docs = vectordb.similarity_search(message, k=3, filter={"category": "general"})

        else:
            docs = vectordb.similarity_search(message, k=3, filter={"role": user["role"]})

        if not docs:
            return {"response": f"No relevant data found for your role: {user['role']}"}

        # -----------------------------
        # Build prompt
        # -----------------------------
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an AI assistant at FinSolve Technologies.

The user has the role: {user['role']}

Use the context below to answer the user's question in a friendly, clear, conversational style.
Summarize naturally; do not just dump bullet points.
If the context is insufficient, say what’s missing and ask 1–2 clarifying questions.

Context:
{context}

Question:
{message}

Answer:
""".strip()

        # -----------------------------
        # Gemini response
        # -----------------------------
        model = genai.GenerativeModel(GEMINI_MODEL)
        result = model.generate_content(prompt)
        llm_answer = (result.text or "").strip()

        return {
            "username": user["username"],
            "role": user["role"],
            "query": message,
            "response": llm_answer if llm_answer else "⚠️ Empty response from Gemini.",
        }

    except Exception as e:
        return {"response": f"Error during chat: {str(e)}"}