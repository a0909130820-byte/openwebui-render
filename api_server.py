import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from query_with_ollama_new import (
    QDRANT_URL as DEFAULT_QDRANT_URL,
    QDRANT_API_KEY as DEFAULT_QDRANT_API_KEY,
    get_embedder,
    retrieve_hits,
)

# ===============================
# FastAPI 基本設定
# ===============================
app = FastAPI(title="CNC Error Code RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 部署測試先開全部；正式可改成 GitHub Pages 網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 系統設定
# ===============================
# 優先讀 Render Environment Variables，沒有才用 query_with_ollama_new.py 裡的預設值
QDRANT_URL = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", DEFAULT_QDRANT_API_KEY)

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "error_codes")

TOP_K = int(os.getenv("TOP_K", "30"))
BOOST = float(os.getenv("BOOST", "0.2"))
RERANK_TOPN = int(os.getenv("RERANK_TOPN", "5"))

# 載入 embedding 模型
embedder = get_embedder(EMB_MODEL)


class AskRequest(BaseModel):
    question: str


def normalize_text(text: str) -> str:
    """整理空白與換行，避免輸出太亂。"""
    text = str(text or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_section(text: str, start: str, end: str | None = None) -> str:
    """
    從 chunk 文字中抽出：
    ERROR MESSAGE
    CAUSE OF ERROR
    ERROR CORRECTION
    """
    text = normalize_text(text)

    if start not in text:
        return "資料片段未提供"

    part = text.split(start, 1)[1]

    if end and end in part:
        part = part.split(end, 1)[0]

    part = normalize_text(part)

    return part if part else "資料片段未提供"


def find_error_code_from_text(text: str) -> str:
    m = re.search(r"[A-Za-z0-9]+-[A-Za-z0-9]+", text or "")
    return m.group(0) if m else "資料片段未提供"


def format_answer(hit: dict) -> str:
    """
    將 Qdrant 檢索到的第一筆最佳資料整理成固定答案格式。
    不使用 Ollama，不會自行補充或亂編。
    """
    text = normalize_text(hit.get("text", "") or hit.get("embedding_text", ""))

    error_code = str(hit.get("error_code", "")).strip()
    if not error_code:
        error_code = find_error_code_from_text(text)

    page = hit.get("page", "資料片段未提供")

    error_message = extract_section(text, "ERROR MESSAGE", "CAUSE OF ERROR")
    cause_of_error = extract_section(text, "CAUSE OF ERROR", "ERROR CORRECTION")
    error_correction = extract_section(text, "ERROR CORRECTION", None)

    return f"""1. 錯誤代碼：
- {error_code}

2. Error message：
- {error_message}

3. Cause of error：
- {cause_of_error}

4. Error correction：
- {error_correction}

5. 頁碼：
- {page}
"""


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC RAG API is running without Ollama",
        "collection": QDRANT_COLLECTION,
        "embedding_model": EMB_MODEL,
        "rerank_model": RERANK_MODEL,
    }


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        return {
            "question": question,
            "answer": "請輸入錯誤代碼或問題。"
        }

    hits = retrieve_hits(
        question=question,
        emb_model_name=EMB_MODEL,
        embedder=embedder,
        qdrant_url=QDRANT_URL,
        qdrant_collection=QDRANT_COLLECTION,
        top_k=TOP_K,
        boost=BOOST,
        rerank_model=RERANK_MODEL,
        rerank_topn=RERANK_TOPN,
    )

    if not hits:
        return {
            "question": question,
            "answer": "查無相關資料。"
        }

    best_hit = hits[0]
    answer = format_answer(best_hit)

    return {
        "question": question,
        "answer": answer,
        "debug": {
            "page": best_hit.get("page", ""),
            "error_code": best_hit.get("error_code", ""),
            "score": best_hit.get("_score", ""),
            "rerank": best_hit.get("_rerank", ""),
            "retrieval": best_hit.get("_retrieval", ""),
            "text_preview": normalize_text(best_hit.get("text", ""))[:500],
        }
    }
