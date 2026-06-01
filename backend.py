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


def collect_images(results: List[dict]) -> List[str]:
    all_images = []

    for payload in results:
        metadata = payload.get("metadata", {}) or {}
        original_metadata = payload.get("original_metadata", {}) or {}

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
# 目的：讓「G02 是什麼」、「G02怎麼用」、「231-E018 怎麼排除」也能命中
# 不使用 sentence-transformers，不增加 Render 負擔
# =========================
def normalize_query_text(text: str) -> str:
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
    q = normalize_query_text(query)

    # 抽出常見 CNC / 警報 / 參數關鍵代碼
    patterns = [
        r"G\d{1,3}(?:\.\d)?",
        r"M\d{1,3}",
        r"INT\d+",
        r"MOT\d+",
        r"OP\d+",
        r"RTEX\d+",
        r"ETHERCAT",
        r"\d{3,4}-[A-Z0-9]{1,6}",
        r"\d{4}",
    ]

    terms = []

    for p in patterns:
        terms.extend(re.findall(p, q, flags=re.IGNORECASE))

    # 抽中文關鍵詞，讓「圓弧插補怎麼用」、「主軸正轉」也能稍微命中
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    terms.extend(chinese_terms)

    # 如果都沒有抽到，就用原問題
    if not terms and q:
        terms.append(q)

    clean_terms = []
    for t in terms:
        t = normalize_query_text(t)
        if t and t not in clean_terms:
            clean_terms.append(t)

    return clean_terms


def payload_to_search_text(payload: dict) -> str:
    metadata = payload.get("metadata", {}) or {}
    original_metadata = payload.get("original_metadata", {}) or {}

    parts = [
        payload.get("source_file", ""),
        payload.get("source_pdf", ""),
        payload.get("manual_type", ""),
        payload.get("page", ""),
        payload.get("title", ""),
        payload.get("section", ""),
        payload.get("major_title", ""),
        payload.get("minor_title", ""),
        payload.get("text", ""),
        payload.get("embedding_text", ""),
        metadata,
        original_metadata,
    ]

    return normalize_query_text(" ".join([str(p) for p in parts]))


def keyword_search(query: str, limit: int = 5):
    results = []
    offset = None
    terms = extract_search_terms(query)
    raw_q = normalize_query_text(query)

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

            # 代碼 / 關鍵詞命中加權
            for term in terms:
                if term and term in search_text:
                    score += 10

            # 原句去空白後如果也命中，補分
            if raw_q in search_text:
                score += 3

            # 輕量中文斷詞：例如「圓弧插補怎麼用」會拆出「圓弧插補怎麼用」本身，
            # 也嘗試用 2 字以上中文字子串做弱命中
            for word in re.findall(r"[\u4e00-\u9fff]{2,}", query):
                word = normalize_query_text(word)
                if word and word in search_text:
                    score += 2

            if score > 0:
                item = dict(payload)
                item["_score"] = score
                results.append(item)

        if offset is None:
            break

    # 依分數排序，避免問「G02 是什麼」時不是 G02 頁排前面
    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # 去重：同一來源同一頁只留一次
    unique = []
    seen = set()

    for r in results:
        metadata = r.get("metadata", {}) or {}
        source = (
            r.get("source_file")
            or r.get("source_pdf")
            or metadata.get("source_pdf")
            or ""
        )
        page = r.get("page") or metadata.get("page") or ""
        text_head = str(r.get("text", ""))[:80]

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
