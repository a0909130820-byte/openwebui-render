import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel
from qdrant_client import QdrantClient


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
# 圖片路徑會直接使用 Qdrant payload / original_metadata 裡的 images
# 例如：/static/images/L2100_programming_p23_2.png
# =========================
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# 環境變數
# Render 請設定：
# QDRANT_URL
# QDRANT_API_KEY
# GEMINI_API_KEY
# QDRANT_COLLECTION  例如：l2100_manuals 或 maintenance_manual_v3
# =========================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "l2100_manuals")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# =========================
# Qdrant
# =========================
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False,
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
        "gemini_model": GEMINI_MODEL,
    }


# =========================
# GPT風格前端 UI
# =========================
@app.get("/ui")
def ui():
    return FileResponse("index.html")


# =========================
# 工具：轉 list
# =========================
def ensure_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


# =========================
# 工具：標準化文字
# 目的：讓 g02 / G02 / G 02 都比較容易命中
# =========================
def normalize_for_search(text: Any) -> str:
    s = str(text or "").lower()
    s = s.replace("－", "-")
    s = re.sub(r"\s+", "", s)
    return s


def build_query_variants(query: str) -> List[str]:
    q = str(query or "").strip()
    q_lower = q.lower()
    q_no_space = normalize_for_search(q)

    variants = {q_lower, q_no_space}

    # g02 / G02 / g 02 這類代碼加強
    m = re.fullmatch(r"([a-zA-Z])\s*0*(\d{1,3})(?:\.(\d))?", q)
    if m:
        letter = m.group(1).lower()
        number = int(m.group(2))
        decimal = m.group(3)
        code = f"{letter}{number:02d}"
        variants.add(code)
        variants.add(f"{letter}{number}")
        variants.add(f"{letter} {number:02d}")
        if decimal:
            variants.add(f"{code}.{decimal}")

    return [v for v in variants if v]


# =========================
# payload 正規化
# 你的 embed 程式目前把 manual_type / codes / images 放在 original_metadata 裡。
# 這裡會同時支援：
# 1. payload 最外層欄位
# 2. original_metadata 裡面的欄位
# 所以不用重新 embedding 到 Qdrant。
# =========================
def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}
    original_metadata = payload.get("original_metadata", {}) or {}

    source_pdf = payload.get("source_pdf") or original_metadata.get("source_pdf") or ""

    source_file = (
        payload.get("source_file")
        or source_pdf
        or original_metadata.get("source_file")
        or ""
    )

    title = (
        payload.get("title")
        or payload.get("minor_title")
        or payload.get("major_title")
        or payload.get("section")
        or original_metadata.get("minor_title")
        or original_metadata.get("major_title")
        or original_metadata.get("section")
        or ""
    )

    manual_type = payload.get("manual_type") or original_metadata.get("manual_type") or ""
    codes = ensure_list(payload.get("codes") or original_metadata.get("codes"))
    images = ensure_list(payload.get("images") or original_metadata.get("images"))
    page = payload.get("page") or original_metadata.get("page") or ""

    # 保險：如果 image_map 裡不是 /static/images 開頭，就補上
    fixed_images = []
    for img in images:
        img = str(img).strip()
        if not img:
            continue
        if img.startswith("http") or img.startswith("/static/"):
            fixed_images.append(img)
        else:
            fixed_images.append(f"/static/images/{img}")

    return {
        **payload,
        "source_file": str(source_file),
        "source_pdf": str(source_pdf),
        "title": str(title),
        "manual_type": str(manual_type),
        "codes": [str(c) for c in codes if str(c).strip()],
        "images": fixed_images,
        "page": page,
        "section": payload.get("section") or original_metadata.get("section") or "",
        "major_title": payload.get("major_title") or original_metadata.get("major_title") or "",
        "minor_title": payload.get("minor_title") or original_metadata.get("minor_title") or "",
        "content_type": payload.get("content_type") or original_metadata.get("content_type") or "",
        "text": payload.get("text", ""),
        "embedding_text": payload.get("embedding_text", ""),
        "original_metadata": original_metadata,
    }


# =========================
# 建立 context
# =========================
def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        codes = r.get("codes", [])
        codes_text = "、".join([str(c) for c in codes]) if isinstance(codes, list) else str(codes)

        context += f"""
【資料 {i}】

來源：
{r.get("source_file", "")}

手冊類型：
{r.get("manual_type", "")}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", "")}

代碼：
{codes_text}

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
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text


# =========================
# 關鍵字搜尋
# 只讀 Qdrant，不讀三個 image_map json。
# 所以不用重新 embedding，只要 Qdrant payload 的 original_metadata 有 images/codes/manual_type 即可。
# =========================
def keyword_search(query: str, limit: int = 5):
    results = []
    offset = None
    variants = build_query_variants(query)

    if not variants:
        return results

    for _ in range(200):
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = normalize_payload(p.payload or {})

            search_fields = [
                payload.get("source_file", ""),
                payload.get("source_pdf", ""),
                payload.get("manual_type", ""),
                payload.get("title", ""),
                payload.get("section", ""),
                payload.get("major_title", ""),
                payload.get("minor_title", ""),
                payload.get("content_type", ""),
                " ".join([str(c) for c in payload.get("codes", [])]),
                payload.get("text", ""),
                payload.get("embedding_text", ""),
            ]

            search_text_raw = " ".join([str(x) for x in search_fields]).lower()
            search_text_compact = normalize_for_search(search_text_raw)

            matched = False
            for v in variants:
                if v in search_text_raw or normalize_for_search(v) in search_text_compact:
                    matched = True
                    break

            if matched:
                results.append(payload)

            if len(results) >= limit:
                return results

        if offset is None:
            break

    return results


# =========================
# 搜尋 API
# =========================

def collect_title_related_images(query: str, results: List[dict], max_images: int = 12) -> List[str]:
    """
    只補抓「同 source_pdf + manual_type + section/minor_title/title」的圖片。
    不用鄰近頁，避免 G09 抓到 G10。

    若使用者問的是 G02 / G09 / M03 這種代碼：
    只允許使用「標題本身有該代碼」的 section/minor_title/title 來補圖。
    """
    images: List[str] = []

    def add_images(payload: Dict[str, Any]):
        for img in payload.get("images", []):
            if img and img not in images:
                images.append(img)

    def valid_heading(value: Any) -> str:
        heading = str(value or "").strip()
        heading_norm = normalize_for_search(heading)
        if not heading_norm:
            return ""
        if heading_norm.isdigit():
            return ""
        if len(heading_norm) < 4:
            return ""
        return heading_norm

    query_codes = extract_codes_from_query(query)
    code_variants = set()

    for code in query_codes:
        code_norm = normalize_for_search(code)
        if code_norm:
            code_variants.add(code_norm)

        m = re.fullmatch(r"([a-zA-Z])\s*0*(\d{1,3})(?:\.(\d))?", str(code).strip())
        if m:
            letter = m.group(1).lower()
            number = int(m.group(2))
            decimal = m.group(3)
            code_variants.add(f"{letter}{number:02d}")
            code_variants.add(f"{letter}{number}")
            if decimal:
                code_variants.add(f"{letter}{number:02d}.{decimal}")

    target_keys = set()

    for r in results:
        source_pdf = str(r.get("source_pdf", "")).strip()
        manual_type = str(r.get("manual_type", "")).strip()

        if not source_pdf:
            continue

        for field in ["section", "minor_title", "title"]:
            heading_norm = valid_heading(r.get(field, ""))

            if not heading_norm:
                continue

            # 代碼查詢時，只有標題包含該代碼才拿來當圖片關聯標題
            if code_variants and not any(cv and cv in heading_norm for cv in code_variants):
                continue

            target_keys.add((source_pdf, manual_type, heading_norm))

    # 找不到可用標題時，保留原本搜尋結果自己的圖片，不做額外擴充
    if not target_keys:
        for r in results:
            add_images(r)
        return images[:max_images]

    offset = None

    for _ in range(200):
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = normalize_payload(p.payload or {})

            if not payload.get("images"):
                continue

            source_pdf = str(payload.get("source_pdf", "")).strip()
            manual_type = str(payload.get("manual_type", "")).strip()

            matched_same_heading = False

            for field in ["section", "minor_title", "title"]:
                heading_norm = valid_heading(payload.get(field, ""))

                if not heading_norm:
                    continue

                if (source_pdf, manual_type, heading_norm) in target_keys:
                    matched_same_heading = True
                    break

            if not matched_same_heading:
                continue

            add_images(payload)

            if len(images) >= max_images:
                return images[:max_images]

        if offset is None:
            break

    return images[:max_images]

@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    try:
        results = keyword_search(query=query, limit=5)

        if not results:
            return {
                "query": query,
                "collection": COLLECTION_NAME,
                "count": 0,
                "answer": f"查無相關資料。請先確認 Render 的 QDRANT_COLLECTION 是否指向正確 collection：{COLLECTION_NAME}",
                "results": [],
                "images": [],
            }

        images = collect_title_related_images(query, results, max_images=12)

        try:
            answer = generate_answer(query, results)
        except Exception as gemini_error:
            answer = (
                f"Gemini 生成失敗：{gemini_error}\n\n"
                f"以下是檢索到的原始資料：\n\n{build_context(results)[:3000]}"
            )

        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": len(results),
            "answer": answer,
            "results": results,
            "images": images,
        }

    except Exception as e:
        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": 0,
            "answer": f"後端錯誤：{e}",
            "results": [],
            "images": [],
        }
