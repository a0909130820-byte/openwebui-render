import os
import re
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

API_KEY = os.getenv("API_KEY", "change-me")
QDRANT_URL = os.getenv("https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODM2Mzk4MDUtYTVmNS00MzUyLWE2NWEtZWNlMWUxNWYxZTE3In0.SStK2mFTKzbEvbWc2r8B2s7TiXE68ETTKrPvmrkiJ7A"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "error_codes")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

OLLAMA_URL = os.getenv("OLLAMA_URL", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

app = FastAPI(title="CNC RAG OpenAI Compatible API")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

embedder = SentenceTransformer(EMBED_MODEL)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "cnc-rag"
    messages: List[ChatMessage]
    temperature: float = 0.2
    stream: bool = False


def check_auth(authorization: str | None):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def extract_error_code(text: str):
    # 支援 231-A043、251-0E2D、1A0-01B8、ABC-12D3E
    pattern = r"\b[A-Za-z0-9]+-[A-Za-z0-9]{3,6}\b"
    m = re.search(pattern, text)
    return m.group(0).upper() if m else None


def exact_search_error_code(error_code: str):
    result, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "error_code",
                    "match": {"value": error_code}
                }
            ]
        },
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    return result


def vector_search(query: str, limit: int = 8):
    query_for_embed = "query: " + query if "bge" in EMBED_MODEL.lower() else query
    vector = embedder.encode(query_for_embed).tolist()

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
        with_payload=True,
    )
    return hits


def build_context_from_points(points):
    chunks = []

    for p in points:
        payload = p.payload or {}

        error_code = payload.get("error_code", "")
        page = payload.get("page", "")
        message = payload.get("ERROR MESSAGE", "")
        cause = payload.get("CAUSE OF ERROR", "")
        correction = payload.get("ERROR CORRECTION", "")
        text = payload.get("text", "")

        chunk = f"""
[page={page}] [error_code={error_code}]
ERROR MESSAGE: {message}
CAUSE OF ERROR: {cause}
ERROR CORRECTION: {correction}
{text}
""".strip()

        chunks.append(chunk)

    return "\n\n---\n\n".join(chunks)


def call_ollama(prompt: str):
    if not OLLAMA_URL:
        return "目前尚未設定 OLLAMA_URL，因此只回傳檢索結果。\n\n" + prompt

    url = OLLAMA_URL.rstrip("/") + "/api/generate"

    r = requests.post(
        url,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120,
    )

    r.raise_for_status()
    return r.json().get("response", "")


def answer_question(user_question: str):
    error_code = extract_error_code(user_question)

    if error_code:
        points = exact_search_error_code(error_code)

        if points:
            context = build_context_from_points(points)
            return f"""已找到錯誤代碼：{error_code}

{context}
"""

    hits = vector_search(user_question, limit=8)
    context = build_context_from_points(hits)

    prompt = f"""
你是 CNC 車床售後服務系統助手。
請只能根據下方資料回答。
如果資料不足，請說「資料不足，無法確認」。
回答要包含：
1. 錯誤代碼
2. 錯誤訊息
3. 可能原因
4. 排除方式
5. 來源頁碼

使用者問題：
{user_question}

檢索資料：
{context}
"""

    return call_ollama(prompt)


@app.get("/")
def root():
    return {"status": "ok", "message": "CNC RAG API is running"}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": "cnc-rag",
                "object": "model",
                "created": 0,
                "owned_by": "local-rag"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatRequest,
    authorization: str | None = Header(default=None)
):
    check_auth(authorization)

    user_question = ""

    for m in reversed(req.messages):
        if m.role == "user":
            user_question = m.content
            break

    if not user_question:
        raise HTTPException(status_code=400, detail="No user message found")

    answer = answer_question(user_question)

    return {
        "id": "chatcmpl-cnc-rag",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }