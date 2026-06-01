import os
import re
import time
from typing import List, Dict, Any

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

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# Render Environment
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "l2100_manuals").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


# =========================
# Clients
# =========================
qdrant = None
if QDRANT_URL:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=False,
        https=True,
        timeout=60,
        check_compatibility=False,
    )

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


class QueryRequest(BaseModel):
    query: str
    manual_type: str = "all"
    use_ollama: bool = True


# =========================
# Basic API
# =========================
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "L2100 CNC AI API is running",
        "collection": COLLECTION_NAME,
        "qdrant_url_set": bool(QDRANT_URL),
        "qdrant_api_key_set": bool(QDRANT_API_KEY),
        "gemini_api_key_set": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.get("/health/qdrant")
def qdrant_health():
    if qdrant is None:
        return {
            "status": "error",
            "collection": COLLECTION_NAME,
            "error": "QDRANT_URL 沒有設定，請到 Render Environment 設定 QDRANT_URL",
        }

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
# Payload helpers
# 支援兩種 payload 結構：
# 1. 舊版：payload["source_file"], payload["page"], payload["codes"], payload["images"]
# 2. 新版：payload["metadata"]["source_pdf"], payload["metadata"]["codes"], payload["metadata"]["images"]
# =========================
def get_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta


def get_original_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("original_metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta


def get_payload_value(payload: Dict[str, Any], key: str, default=""):
    meta = get_meta(payload)
    original = get_original_meta(payload)

    if key in payload and payload.get(key) not in [None, ""]:
        return payload.get(key)

    if key in meta and meta.get(key) not in [None, ""]:
        return meta.get(key)

    if key in original and original.get(key) not in [None, ""]:
        return original.get(key)

    # 常用欄位名稱對應
    if key == "source_file":
        return (
            payload.get("source_file")
            or payload.get("source_pdf")
            or meta.get("source_pdf")
            or original.get("source_pdf")
            or default
        )

    if key == "title":
        return (
            payload.get("title")
            or payload.get("major_title")
            or meta.get("major_title")
            or original.get("major_title")
            or default
        )

    return default


def normalize(text) -> str:
    return (
        str(text)
        .upper()
        .replace(" ", "")
        .replace("　", "")
        .replace("\n", "")
        .replace("\\N", "")
    )


def extract_keywords(query: str) -> List[str]:
    q = normalize(query)

    keys = re.findall(
        r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|ETHERCAT|參數\d+|\d{3,4}[-－][A-Z0-9]{1,6}|\d{4}",
        q,
        flags=re.IGNORECASE
    )

    out = []
    for k in keys:
        k = normalize(k)
        if k and k not in out:
            out.append(k)

    if not out and q:
        out = [q]

    return out


def manual_type_of(payload: Dict[str, Any]) -> str:
    return str(
        get_payload_value(payload, "manual_type", "")
    ).strip()


def codes_of(payload: Dict[str, Any]):
    codes = (
        payload.get("codes")
        or get_meta(payload).get("codes")
        or get_original_meta(payload).get("codes")
        or []
    )

    if isinstance(codes, list):
        return [str(c) for c in codes]

    if isinstance(codes, str):
        return [codes]

    return []


def images_of(payload: Dict[str, Any]) -> List[str]:
    image_sources = [
        payload.get("images", []),
        get_meta(payload).get("images", []),
        get_original_meta(payload).get("images", []),
    ]

    images = []

    for imgs in image_sources:
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


def collect_images(results: List[dict]) -> List[str]:
    all_images = []

    for payload in results:
        for img in images_of(payload):
            if img not in all_images:
                all_images.append(img)

    return all_images


# =========================
# 搜尋方式：整合 backend1111 的 keyword score
# =========================
def build_search_text(payload: Dict[str, Any]) -> str:
    parts = [
        get_payload_value(payload, "source_file", ""),
        get_payload_value(payload, "manual_type", ""),
        get_payload_value(payload, "page", ""),
        get_payload_value(payload, "title", ""),
        get_payload_value(payload, "section", ""),
        get_payload_value(payload, "major_title", ""),
        get_payload_value(payload, "minor_title", ""),
        " ".join(codes_of(payload)),
        payload.get("text", ""),
        payload.get("embedding_text", ""),
    ]

    return normalize(" ".join([str(p) for p in parts if p is not None]))


def keyword_search(query: str, manual_type: str = "all", limit: int = 10):
    if qdrant is None:
        raise RuntimeError("QDRANT_URL 沒有設定，後端無法連線 Qdrant")

    keys = extract_keywords(query)
    results = []
    offset = None

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

            if manual_type != "all":
                if manual_type_of(payload) != manual_type:
                    continue

            search_text = build_search_text(payload)

            score = 0

            for key in keys:
                if key in search_text:
                    score += 10

            # 如果沒有抓到代碼，退回一般子字串搜尋
            if score == 0:
                q = normalize(query)
                if q and q in search_text:
                    score += 1

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
        uid = (
            str(get_payload_value(r, "source_file", "")),
            str(get_payload_value(r, "page", "")),
            str(r.get("text", ""))[:120],
        )

        if uid in seen:
            continue

        seen.add(uid)
        unique.append(r)

        if len(unique) >= limit:
            break

    return unique


# =========================
# 回答組裝
# =========================
def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        source_file = get_payload_value(r, "source_file", "")
        page = get_payload_value(r, "page", "")
        title = get_payload_value(r, "title", "")
        section = get_payload_value(r, "section", "")
        manual_type = get_payload_value(r, "manual_type", "")
        codes = "、".join(codes_of(r))
        text = str(r.get("text", "")).strip()

        context += f"""
【資料 {i}】

手冊類型：
{manual_type}

來源：
{source_file}

頁碼：
{page}

標題：
{title}

章節：
{section}

代碼：
{codes}

內容：
{text[:2500]}
"""

    return context


def build_sources(results: List[dict]) -> List[dict]:
    sources = []
    seen = set()

    for r in results:
        source_file = get_payload_value(r, "source_file", "")
        page = get_payload_value(r, "page", "")
        title = get_payload_value(r, "title", "")

        key = (str(source_file), str(page))

        if source_file and page and key not in seen:
            seen.add(key)
            sources.append({
                "source_file": source_file,
                "page": page,
                "title": title,
            })

    return sources


def generate_answer(query: str, results: List[dict]) -> str:
    context = build_context(results)

    if not gemini_client:
        return "【未設定 Gemini API Key，改用手冊原文模式】\n\n" + context[:4000]

    prompt = f"""
你是 L2100 CNC 車床手冊 AI 助理。

只能根據以下手冊內容回答，不可以幻想不存在的資訊。
若資料不足，請說「目前資料中沒有找到足夠資訊」。

使用者問題：
{query}

手冊內容：
{context}

請使用繁體中文整理：
1. 查詢重點
2. 功能/錯誤說明
3. 使用或排除建議
4. 來源頁碼
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as e:
        print("Gemini 失敗：", e)
        return (
            f"【Gemini 生成失敗，改用手冊原文模式】\n"
            f"錯誤：{e}\n\n"
            f"{context[:5000]}"
        )


@app.post("/search")
def search(req: QueryRequest):
    start_time = time.time()
    query = req.query.strip()

    try:
        results = keyword_search(
            query=query,
            manual_type=req.manual_type,
            limit=10
        )

        elapsed = round(time.time() - start_time, 2)
        images = collect_images(results)

        if not results:
            return {
                "query": query,
                "collection": COLLECTION_NAME,
                "count": 0,
                "elapsed": elapsed,
                "answer": "查無相關資料，請換一個關鍵字。",
                "results": [],
                "sources": [],
                "images": images,
            }

        answer = generate_answer(query, results)

        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": len(results),
            "elapsed": elapsed,
            "answer": answer,
            "results": results,
            "sources": build_sources(results),
            "images": images,
        }

    except Exception as e:
        print("Search Error:", e)

        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": 0,
            "elapsed": round(time.time() - start_time, 2),
            "answer": f"後端錯誤：{e}",
            "results": [],
            "sources": [],
            "images": [],
        }
