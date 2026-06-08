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
    """
    圖片抓取邏輯：
    1. 先抓搜尋結果本身 payload / metadata / original_metadata 裡的 images
    2. 再依照搜尋結果的 source_pdf + page，到 Qdrant 裡補抓同一手冊、同一頁的所有圖片
    3. 不只抓第一張；只要該頁 metadata 有圖片，就全部帶出來
    """
    all_images = []

    def ensure_list(value):
        if not value:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def normalize_image_path(img):
        img = str(img).strip()

        if not img:
            return ""

        if img.startswith("/static/") or img.startswith("http"):
            return img

        return "/static/images/" + img.lstrip("/")

    def get_meta(payload):
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        original_metadata = payload.get("original_metadata", {}) or {}
        if not isinstance(original_metadata, dict):
            original_metadata = {}

        return metadata, original_metadata

    def get_source(payload):
        metadata, original_metadata = get_meta(payload)

        return (
            payload.get("source_file")
            or payload.get("source_pdf")
            or metadata.get("source_pdf")
            or metadata.get("source_file")
            or original_metadata.get("source_pdf")
            or original_metadata.get("source_file")
            or ""
        )

    def get_page(payload):
        metadata, original_metadata = get_meta(payload)

        return (
            payload.get("page")
            or metadata.get("page")
            or original_metadata.get("page")
            or ""
        )

    def add_images_from_payload(payload):
        metadata, original_metadata = get_meta(payload)

        image_sources = [
            payload.get("images", []),
            metadata.get("images", []),
            original_metadata.get("images", []),
        ]

        for imgs in image_sources:
            for img in ensure_list(imgs):
                img = normalize_image_path(img)

                if img and img not in all_images:
                    all_images.append(img)

    # 先收集搜尋結果本身的圖片，並記錄命中資料的來源與頁碼
    target_pages = set()

    for payload in results:
        add_images_from_payload(payload)

        source = str(get_source(payload)).strip()
        page = str(get_page(payload)).strip()

        if source and page:
            target_pages.add((source, page))

    # 再補抓 Qdrant 裡同一手冊、同一頁的其他圖片
    # 這樣如果某個 chunk 本身沒有 images，但同頁其他 chunk 有圖片，也會一起抓到
    if target_pages:
        try:
            offset = None

            for _ in range(100):
                points, offset = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for point in points:
                    payload = point.payload or {}

                    source = str(get_source(payload)).strip()
                    page = str(get_page(payload)).strip()

                    if (source, page) in target_pages:
                        add_images_from_payload(payload)

                if offset is None:
                    break

        except Exception as e:
            print("補抓同頁圖片失敗：", e)

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
# 讓「G02是什麼」「G02怎麼用」「M03功能」可以先抽出 G02 / M03 再搜尋
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
    """
    搜尋詞抽取邏輯：
    1. 不把中文拆成單字或很多小詞
    2. 不另外硬抽 G02/M03/GMC800/EIO4832 這種固定代碼規則
    3. 只使用客戶原本輸入的問題，以及問題中連續出現的中文/英文/數字片段
    4. 例如：
       -「連接器腳位定義」→ 只用「連接器腳位定義」
       -「G02是什麼」→ 用「G02是什麼」和「G02」
       -「Slave I/O 介面說明」→ 用「SLAVEIO介面說明」和「SLAVEIO」、「介面說明」
    """
    raw = str(query).strip()
    q = normalize_search_text(raw)

    terms = []

    # 1. 完整問題一定放第一個
    if q:
        terms.append(q)

    # 2. 抽「連續英數」片段，不限定格式
    #    這樣 G02、M03、GMC800、EIO4832 都會自然被抓到，
    #    但不是靠寫死特定代碼規則。
    alnum_parts = re.findall(r"[A-Za-z0-9]+", raw)
    for part in alnum_parts:
        part = normalize_search_text(part)
        if len(part) >= 2:
            terms.append(part)

    # 3. 抽「連續中文」片段，不拆成小詞
    chinese_parts = re.findall(r"[\u4e00-\u9fff]+", raw)
    for part in chinese_parts:
        part = normalize_search_text(part)
        if len(part) >= 2:
            terms.append(part)

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

            # 重點：G02是什麼 → terms 會有 G02
            for term in terms:
                if term and term in search_text:
                    score += 10

            # 如果原句完整命中，也加分
            if raw_q and raw_q in search_text:
                score += 3

            # 中文詞弱命中
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

    # 分數高的排前面
    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # 去重
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
            or ""
        )

        page = (
            r.get("page")
            or metadata.get("page")
            or ""
        )

        text_head = str(r.get("text", ""))[:100]
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
