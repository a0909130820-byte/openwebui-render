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

if os.path.exists("static"):
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static"
    )


QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "l2100_manuals")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)


gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC L2100 Manual AI API is running",
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


def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {}) or {}
        original_metadata = r.get("original_metadata", {}) or {}

        codes = (
            r.get("codes")
            or metadata.get("codes")
            or original_metadata.get("codes")
            or []
        )

        if isinstance(codes, list):
            codes_text = "、".join([str(c) for c in codes])
        else:
            codes_text = str(codes)

        source_file = (
            r.get("source_file")
            or r.get("source_pdf")
            or metadata.get("source_pdf")
            or original_metadata.get("source_pdf")
            or ""
        )

        page = (
            r.get("page")
            or metadata.get("page")
            or original_metadata.get("page")
            or ""
        )

        title = (
            r.get("title")
            or r.get("major_title")
            or metadata.get("major_title")
            or original_metadata.get("major_title")
            or ""
        )

        section = (
            r.get("section")
            or metadata.get("section")
            or original_metadata.get("section")
            or ""
        )

        text = str(r.get("text", "")).strip()

        context += f"""
【資料 {i}】

來源：
{source_file}

頁碼：
{page}

標題：
{title}

章節：
{section}

代碼：
{codes_text}

內容：
{text[:2500]}
"""

    return context



def generate_answer(query: str, results: List[dict]) -> str:
    context = build_context(results)

    if not gemini_client:
        return "查到手冊資料，但未設定 Gemini API Key。\n\n" + context[:3000]

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
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text



# =========================
# 輕量自然語句搜尋工具
# 讓「G02是什麼」「G02怎麼用」「M03功能」可以先抽出 G02 / M03 再搜尋
# =========================




# =========================
# 搜尋輔助：中文不拆弱詞、標題/章節優先
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
        .replace("/", "")
        .replace("_", "")
        .replace("-", "")
    )


def extract_search_terms(query: str):
    """
    不把中文拆成很多小詞。
    只保留：
    1. 完整問題
    2. 使用者輸入中的連續英數片段，例如 G02、GMC800、EIO4832
    3. 使用者輸入中的連續中文片段，例如 連接器腳位定義
    """
    raw = str(query).strip()
    q = normalize_search_text(raw)

    terms = []

    if q:
        terms.append(q)

    # 連續英數，不限定固定代碼格式
    for part in re.findall(r"[A-Za-z0-9]+", raw):
        part = normalize_search_text(part)
        if len(part) >= 2:
            terms.append(part)

    # 連續中文，不拆成小詞
    for part in re.findall(r"[\u4e00-\u9fff]+", raw):
        part = normalize_search_text(part)
        if len(part) >= 2:
            terms.append(part)

    cleaned = []
    for term in terms:
        term = normalize_search_text(term)
        if term and term not in cleaned:
            cleaned.append(term)

    return cleaned


def get_search_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def build_title_search_text(payload: dict) -> str:
    metadata = get_search_metadata(payload)

    parts = [
        payload.get("title", ""),
        payload.get("section", ""),
        payload.get("major_title", ""),
        payload.get("minor_title", ""),
        metadata.get("title", ""),
        metadata.get("section", ""),
        metadata.get("major_title", ""),
        metadata.get("minor_title", ""),
    ]

    return normalize_search_text(" ".join([str(p) for p in parts if p is not None]))


def build_body_search_text(payload: dict) -> str:
    metadata = get_search_metadata(payload)

    parts = [
        payload.get("source_file", ""),
        payload.get("source_pdf", ""),
        payload.get("manual_type", ""),
        payload.get("page", ""),
        payload.get("error_code", ""),
        payload.get("codes", ""),
        payload.get("text", ""),
        payload.get("embedding_text", ""),
        metadata,
    ]

    return normalize_search_text(" ".join([str(p) for p in parts if p is not None]))


def collect_images(results: List[dict]) -> List[str]:
    """
    圖片只跟命中的資料走：
    只讀 payload.images / metadata.images / original_metadata.images。
    不再用 page 去 image_map 補圖，避免三本 PDF 同頁碼混圖。
    """
    all_images = []

    for payload in results:
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        original_metadata = payload.get("original_metadata", {}) or {}
        if not isinstance(original_metadata, dict):
            original_metadata = {}

        image_sources = [
            payload.get("images", []),
            metadata.get("images", []),
            original_metadata.get("images", []),
        ]

        for imgs in image_sources:
            if not imgs:
                continue

            if isinstance(imgs, str):
                imgs = [imgs]

            for img in imgs:
                if not img:
                    continue

                img = str(img).strip()

                if not img.startswith("/static/") and not img.startswith("http"):
                    img = "/static/images/" + img.lstrip("/")

                if img not in all_images:
                    all_images.append(img)

    return all_images


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

            title_text = build_title_search_text(payload)
            body_text = build_body_search_text(payload)

            score = 0
            title_hits = 0
            body_hits = 0

            # 完整問題命中標題/章節：最高
            if raw_q and raw_q in title_text:
                score += 120

            # 完整問題命中內文：次高
            if raw_q and raw_q in body_text:
                score += 50

            for term in terms:
                if not term or term == raw_q:
                    continue

                # 使用者輸入中的連續片段命中標題/章節
                if term in title_text:
                    title_hits += 1
                    score += 45

                # 使用者輸入中的連續片段命中內文
                elif term in body_text:
                    body_hits += 1
                    score += 18

            # 多個連續片段同時命中標題，加權
            if title_hits >= 2:
                score += title_hits * 25

            # 多個片段只在內文命中，保留但較低
            if body_hits >= 2:
                score += body_hits * 8

            # 避免只命中很弱的單一片段就進結果
            if len(terms) >= 2 and title_hits == 0 and body_hits <= 1 and score < 40:
                score = 0

            if score > 0:
                item = dict(payload)
                item["_score"] = score
                results.append(item)

        if offset is None:
            break

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # 相關度門檻：只保留接近第一名的結果，避免弱相關頁混進來
    if results:
        best_score = results[0].get("_score", 0)
        min_score = best_score * 0.55
        results = [
            r for r in results
            if r.get("_score", 0) >= min_score
        ]

    unique = []
    seen = set()

    for r in results:
        metadata = r.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source = (
            r.get("source_file")
            or r.get("source_pdf")
            or metadata.get("source_pdf")
            or metadata.get("source_file")
            or ""
        )

        page = (
            r.get("page")
            or metadata.get("page")
            or ""
        )

        text_head = str(r.get("text", ""))[:120]
        key = (str(source), str(page), text_head)

        if key in seen:
            continue

        seen.add(key)
        unique.append(r)

        if len(unique) >= limit:
            return unique

    return unique

@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    try:
        results = keyword_search(
            query=query,
            limit=5
        )

        images = collect_images(results)

        if not results:
            return {
                "query": query,
                "collection": COLLECTION_NAME,
                "count": 0,
                "answer": "查無相關資料，請換一個關鍵字。",
                "results": [],
                "images": images
            }

        try:
            answer = generate_answer(query, results)
            answer = f"查到 {len(results)} 筆手冊資料\n\n" + answer

        except Exception as gemini_error:
            fallback_context = build_context(results)
            answer = (
                f"查到 {len(results)} 筆手冊資料\n\n"
                f"Gemini 生成失敗：{gemini_error}\n\n"
                f"以下先顯示檢索到的原始資料：\n\n"
                f"{fallback_context[:4000]}"
            )

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
