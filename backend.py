import os
import re
import json
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

# 圖片對照表：支援 image_map.json 格式
# 例如：{"8": ["L2100_programming_p8_2.png"]}
IMAGE_MAP_PATH = os.getenv("IMAGE_MAP_PATH", "image_map.json")

if os.path.exists(IMAGE_MAP_PATH):
    with open(IMAGE_MAP_PATH, "r", encoding="utf-8") as f:
        IMAGE_MAP = json.load(f)
else:
    IMAGE_MAP = {}


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
        "image_map_path": IMAGE_MAP_PATH,
        "image_map_loaded": bool(IMAGE_MAP),
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
# Payload / 圖片工具
# 支援新版 payload:
# {
#   "metadata": {
#       "page": 8,
#       "source_pdf": "...",
#       "section": "...",
#       "images": [...]
#   },
#   "text": "..."
# }
# 也支援舊版 payload 直接放 images/page/source_file
# =========================
def get_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata
    return {}


def get_payload_value(payload: dict, key: str, default=""):
    metadata = get_metadata(payload)

    if key in payload and payload.get(key) not in [None, ""]:
        return payload.get(key)

    if key in metadata and metadata.get(key) not in [None, ""]:
        return metadata.get(key)

    if key == "source_file":
        return (
            payload.get("source_file")
            or payload.get("source_pdf")
            or metadata.get("source_pdf")
            or default
        )

    if key == "title":
        return (
            payload.get("title")
            or payload.get("major_title")
            or metadata.get("major_title")
            or default
        )

    return default


def normalize_image_url(img: str) -> str:
    img = str(img).strip()

    if not img:
        return ""

    if img.startswith("http"):
        return img

    if img.startswith("/static/"):
        return img

    return "/static/images/" + img.lstrip("/")


def get_images_from_payload(payload: dict):
    metadata = get_metadata(payload)

    images = []

    # 1. 優先讀 Qdrant payload 裡的 images / metadata.images
    image_sources = [
        payload.get("images", []),
        metadata.get("images", []),
    ]

    for imgs in image_sources:
        if not imgs:
            continue

        if isinstance(imgs, str):
            imgs = [imgs]

        for img in imgs:
            url = normalize_image_url(img)
            if url and url not in images:
                images.append(url)

    # 2. 如果 payload 沒有 images，就用 image_map.json 按 page 補圖
    if not images:
        page = get_payload_value(payload, "page", "")
        page_key = str(page)

        if page_key in IMAGE_MAP:
            for img in IMAGE_MAP[page_key]:
                url = normalize_image_url(img)
                if url and url not in images:
                    images.append(url)

    return images


def collect_image_objects(results: list):
    """
    回傳給新版 index.html 的格式：
    [
      {
        "url": "/static/images/xxx.png",
        "source_file": "...",
        "page": 8,
        "section": "..."
      }
    ]
    """
    all_images = []
    seen = set()

    for payload in results:
        source_file = get_payload_value(payload, "source_file", "未知手冊")
        page = get_payload_value(payload, "page", "未知頁")
        section = get_payload_value(payload, "section", "")

        for url in get_images_from_payload(payload):
            if url in seen:
                continue

            seen.add(url)

            all_images.append({
                "url": url,
                "source_file": source_file,
                "page": page,
                "section": section,
            })

    return all_images


# =========================
# 建立 context
# =========================
def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        metadata = get_metadata(r)

        codes = (
            r.get("codes")
            or metadata.get("codes")
            or []
        )

        if isinstance(codes, list):
            codes_text = "、".join([str(c) for c in codes])
        else:
            codes_text = str(codes)

        context += f"""
【資料 {i}】

來源：
{get_payload_value(r, "source_file", "")}

頁碼：
{get_payload_value(r, "page", "")}

標題：
{get_payload_value(r, "title", "")}

章節：
{get_payload_value(r, "section", "")}

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

# =========================
# 輕量自然語句搜尋工具
# 讓「G02是什麼」「G02 怎麼用」「M03功能」「231-E018怎麼排除」可以先抽出代碼再搜尋
# 不使用 sentence-transformers，不增加 Render 負擔
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

    # 中文關鍵詞，例如：圓弧插補、主軸正轉、螺紋切削
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    terms.extend(chinese_terms)

    if not terms and q:
        terms.append(q)

    cleaned = []
    for term in terms:
        term = normalize_search_text(term)
        if term and term not in cleaned:
            cleaned.append(term)

    return cleaned


def payload_to_search_text(payload: dict) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    original_metadata = payload.get("original_metadata", {})
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

            # 主要：G02是什麼 → 抽出 G02 命中
            for term in terms:
                if term and term in search_text:
                    score += 10

            # 原句剛好命中時補分
            if raw_q and raw_q in search_text:
                score += 3

            # 中文詞弱命中，例如「圓弧插補怎麼用」
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

    # 去重：同一來源、同一頁、相同文字開頭只留一次
    unique = []
    seen = set()

    for r in results:
        metadata = r.get("metadata", {})
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

        # 回傳新版 index.html 可讀的圖片物件格式
        images = collect_image_objects(results)

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
