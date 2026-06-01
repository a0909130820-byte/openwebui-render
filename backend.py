import os
import re
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qdrant_client import QdrantClient
from google import genai


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# static 圖片資料夾
# =========================
if os.path.exists("static"):
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static"
    )


# =========================
# 環境變數
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Render Environment 建議設定：
# QDRANT_COLLECTION=l2100_manuals
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "l2100_manuals")


# =========================
# Qdrant
# =========================
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)


# =========================
# Gemini
# =========================
gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# Request Model
# =========================
class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


# =========================
# API 首頁
# =========================
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC L2100 Manual AI API is running",
        "collection": COLLECTION_NAME,
        "qdrant_url_set": bool(QDRANT_URL),
        "qdrant_api_key_set": bool(QDRANT_API_KEY),
        "gemini_api_key_set": bool(GEMINI_API_KEY),
    }


# =========================
# GPT風格前端 UI
# =========================
@app.get("/ui")
def ui():
    return FileResponse("index.html")


# =========================
# 健康檢查：確認 Qdrant 是否連線
# =========================
@app.get("/health/qdrant")
def qdrant_health():
    try:
        info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "points_count": getattr(info, "points_count", None),
            "vectors_count": getattr(info, "vectors_count", None),
        }
    except Exception as e:
        return {
            "status": "error",
            "collection": COLLECTION_NAME,
            "error": str(e),
        }


# =========================
# 建立 context
# =========================
def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        codes = r.get("codes", [])
        if isinstance(codes, list):
            codes_text = "、".join([str(c) for c in codes])
        else:
            codes_text = str(codes)

        context += f"""
【資料 {i}】

來源：
{r.get("source_file", r.get("source_pdf", ""))}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", r.get("major_title", ""))}

章節：
{r.get("section", "")}

代碼：
{codes_text}

錯誤代碼：
{r.get("error_code", "")}

內容：
{str(r.get("text", ""))[:2500]}
"""

    return context


# =========================
# Gemini 回答
# =========================
def generate_answer(query: str, results: List[dict]) -> str:
    context = build_context(results)

    if not gemini_client:
        return context[:3000]

    prompt = f"""
你是 CNC L2100 車床技術手冊 AI 助理。

請只能根據下方資料回答。
不要自己亂猜。

如果資料不足，請直接說：
「目前資料中沒有找到足夠資訊」。

======================

使用者問題：
{query}

======================

檢索到的手冊資料：

{context}

======================

請使用繁體中文回答。

格式：

1. 查詢重點
2. 功能/錯誤說明
3. 操作建議
4. 來源頁碼
"""

    response = gemini_client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt
    )

    return response.text


# =========================
# 關鍵字搜尋
# 重點：不限定 payload 欄位，整個 payload 都拿來比對
# 這樣 codes / error_code / text / title / page 都能搜尋
# =========================

def normalize_search_text(text: str) -> str:
    return (
        str(text)
        .upper()
        .replace(" ", "")
        .replace("　", "")
        .replace("\n", "")
        .replace("\\N", "")
        .replace("－", "-")
    )


def extract_search_terms(query: str):
    q = normalize_search_text(query)

    patterns = [
        r"G\d{1,3}(?:\.\d)?",
        r"M\d{1,3}",
        r"EIO\d+",
        r"IO\d+",
        r"INT\d+",
        r"MOT\d+",
        r"OP\d+",
        r"RTEX\d+",
        r"ETHERCAT",
        r"\d{3,4}-[A-Z0-9]{1,6}",
        r"\d{4}",
    ]

    terms = []

    for pattern in patterns:
        terms.extend(re.findall(pattern, q, flags=re.IGNORECASE))

    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", query))

    if not terms and q:
        terms.append(q)

    cleaned = []
    for term in terms:
        term = normalize_search_text(term)
        if term and term not in cleaned:
            cleaned.append(term)

    return cleaned


def payload_to_search_text(payload: dict) -> str:
    metadata = payload.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    original_metadata = payload.get("original_metadata", {}) or {}
    if not isinstance(original_metadata, dict):
        original_metadata = {}

    parts = [
        payload.get("source_file", ""),
        payload.get("source_pdf", ""),
        payload.get("manual_type", ""),
        payload.get("page", ""),
        payload.get("title", ""),
        payload.get("section", ""),
        payload.get("major_title", ""),
        payload.get("minor_title", ""),
        payload.get("error_code", ""),
        payload.get("text", ""),
        payload.get("embedding_text", ""),
        metadata,
        original_metadata,
    ]

    return normalize_search_text(" ".join([str(p) for p in parts if p is not None]))


def payload_images(payload: dict):
    metadata = payload.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    images = []

    # 只拿這筆資料自己的 images / metadata.images
    # 不拿其他頁、不用 image_map 補圖，避免錯圖
    for imgs in [payload.get("images", []), metadata.get("images", [])]:
        if not imgs:
            continue

        if isinstance(imgs, str):
            imgs = [imgs]

        for img in imgs:
            img = str(img).strip()
            if not img:
                continue

            if not img.startswith("/static/") and not img.startswith("http"):
                img = "/static/images/" + img.lstrip("/")

            if img not in images:
                images.append(img)

    return images


def keyword_search(query: str, limit: int = 5):
    results = []
    offset = None

    raw_query = str(query).strip()
    raw_q = normalize_search_text(raw_query)
    terms = extract_search_terms(raw_query)

    if not raw_q:
        return results

    for _ in range(100):
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for p in points:
            payload = p.payload or {}
            search_text = payload_to_search_text(payload)

            score = 0

            # 例如 G02是什麼 → 抽出 G02；EIO4832 I/O介面說明 → 抽出 EIO4832
            for term in terms:
                if term and term in search_text:
                    score += 10

            if raw_q and raw_q in search_text:
                score += 3

            for word in re.findall(r"[\u4e00-\u9fff]{2,}", raw_query):
                word = normalize_search_text(word)
                if word and word in search_text:
                    score += 2

            if score > 0:
                item = dict(payload)
                item["_score"] = score
                results.append(item)

        if offset is None:
            break

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    unique = []
    seen = set()

    for r in results:
        metadata = r.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source = r.get("source_file") or r.get("source_pdf") or metadata.get("source_pdf") or ""
        page = r.get("page") or metadata.get("page") or ""
        text_head = str(r.get("text", ""))[:100]

        key = (str(source), str(page), text_head)

        if key in seen:
            continue

        seen.add(key)
        unique.append(r)

        if len(unique) >= limit:
            return unique

    return unique

# =========================
# 搜尋 API
# =========================
@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    try:
        results = keyword_search(
            query=query,
            limit=5
        )

        if not results:
            return {
                "query": query,
                "collection": COLLECTION_NAME,
                "count": 0,
                "answer": "查無相關資料，請換一個關鍵字。",
                "results": [],
                "images": []
            }

        images = []

        # 只顯示最相關前 2 筆資料的圖片，避免弱相關資料混入錯圖
        for payload in results[:2]:
            for img in payload_images(payload):
                if img not in images:
                    images.append(img)

        try:
            answer = generate_answer(
                query,
                results
            )

        except Exception as gemini_error:
            answer = f"Gemini 生成失敗：{gemini_error}"

        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": len(results),
            "answer": answer,
            "results": results,
            "images": images
        }

    except Exception as e:
        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": 0,
            "answer": f"後端錯誤：{e}",
            "results": [],
            "images": []
        }
