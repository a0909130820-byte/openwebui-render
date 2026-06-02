import os
import re
from typing import Any, Dict, List, Tuple

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
# =========================
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# 環境變數
# Render 請設定：
# QDRANT_URL
# QDRANT_API_KEY
# GEMINI_API_KEY
# QDRANT_COLLECTION
# GEMINI_MODEL
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


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC L2100 Manual AI API is running",
        "collection": COLLECTION_NAME,
        "gemini_model": GEMINI_MODEL,
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


# =========================
# 基本工具
# =========================
def ensure_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_for_search(text: Any) -> str:
    """
    搜尋用正規化：
    - 大小寫不敏感
    - 移除空白
    - 全形/半形常見符號統一
    """
    s = str(text or "").lower()
    s = s.replace("－", "-").replace("–", "-").replace("—", "-")
    s = s.replace("：", ":").replace("，", ",").replace("。", ".")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s


def extract_codes_from_query(query: str) -> List[str]:
    """
    從整句問題裡抽代碼。
    例如：
    - G02 是什麼 -> G02
    - 231-E018 怎麼排除 -> 231-E018
    - RTEX012 -> RTEX012 / RTEX 012
    """
    q = str(query or "")
    found = []

    patterns = [
        r"\b[GM]\s*0*\d{1,3}(?:\.\d)?\b",
        r"\b\d{3,4}[-－][A-Za-z0-9]{1,8}\b",
        r"\b(?:INT|MOT|OP|RTEX)\s*0*\d{1,4}\b",
        r"\bP\s*0*\d{1,5}\b",
        r"\bALM\s*0*\d{1,5}\b",
    ]

    for pat in patterns:
        for m in re.findall(pat, q, flags=re.IGNORECASE):
            found.append(str(m).strip())

    return list(dict.fromkeys(found))


def build_query_variants(query: str) -> List[str]:
    q = str(query or "").strip()
    variants = set()

    if not q:
        return []

    # 原句
    variants.add(q.lower())
    variants.add(normalize_for_search(q))

    # 常見問句雜訊移除，讓「G02 是什麼」也能命中「G02」
    cleaned = q
    for noise in [
        "是什麼", "是甚麼", "為什麼", "怎麼用", "如何用", "怎麼使用", "如何使用",
        "說明", "介紹", "查詢", "幫我查", "請問", "功能", "意思", "原因", "處理",
        "排除", "方法", "步驟", "？", "?", "。"
    ]:
        cleaned = cleaned.replace(noise, " ")
    cleaned = cleaned.strip()
    if cleaned:
        variants.add(cleaned.lower())
        variants.add(normalize_for_search(cleaned))

    # 從句子中抽出 G/M/警報代碼
    for code_raw in extract_codes_from_query(q):
        c = code_raw.strip()
        variants.add(c.lower())
        variants.add(normalize_for_search(c))

        m = re.fullmatch(r"([a-zA-Z])\s*0*(\d{1,3})(?:\.(\d))?", c)
        if m:
            letter = m.group(1).lower()
            number = int(m.group(2))
            decimal = m.group(3)
            code2 = f"{letter}{number:02d}"
            variants.add(code2)
            variants.add(f"{letter}{number}")
            variants.add(f"{letter} {number:02d}")
            if decimal:
                variants.add(f"{code2}.{decimal}")

        m2 = re.fullmatch(r"(int|mot|op|rtex)\s*0*(\d{1,4})", c, flags=re.IGNORECASE)
        if m2:
            prefix = m2.group(1).lower()
            num = int(m2.group(2))
            variants.add(f"{prefix}{num}")
            variants.add(f"{prefix}{num:03d}")
            variants.add(f"{prefix} {num}")
            variants.add(f"{prefix} {num:03d}")

    # 中文同義詞補強
    synonym_pairs = [
        ("插補", "插值"),
        ("插值", "插補"),
        ("順時針", "順時鐘"),
        ("順時鐘", "順時針"),
        ("逆時針", "逆時鐘"),
        ("逆時鐘", "逆時針"),
        ("錯誤", "警報"),
        ("警報", "錯誤"),
        ("故障", "異常"),
        ("異常", "故障"),
        ("補正", "補償"),
        ("補償", "補正"),
    ]

    for a, b in synonym_pairs:
        if a in q:
            variants.add(q.replace(a, b).lower())
            variants.add(normalize_for_search(q.replace(a, b)))
        if a in cleaned:
            variants.add(cleaned.replace(a, b).lower())
            variants.add(normalize_for_search(cleaned.replace(a, b)))

    # 中文關鍵詞切片：避免整句完全不命中
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    for term in chinese_terms:
        if term not in ["請問", "幫我", "如何", "怎麼", "什麼", "甚麼"]:
            variants.add(term.lower())
            variants.add(normalize_for_search(term))
            for a, b in synonym_pairs:
                if a in term:
                    variants.add(term.replace(a, b).lower())
                    variants.add(normalize_for_search(term.replace(a, b)))

    return [v for v in variants if v]


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    同時支援：
    1. payload 最外層欄位
    2. original_metadata 裡面的欄位

    這樣不用重新 embedding。
    """
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


def generate_answer(query: str, results: List[dict]) -> str:
    context = build_context(results)

    if not gemini_client:
        return context[:3000]

    prompt = f"""
你是 CNC L2100 車床技術手冊 AI 助理。

請只能根據下方資料回答，不要自己亂猜。
如果資料不足，請直接說：「目前資料中沒有找到足夠資訊」。

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
# 搜尋排序
# =========================
def detect_query_intent(query: str) -> str:
    q = normalize_for_search(query)

    # 明確的警報/錯誤/維修詞優先
    if any(k in q for k in ["警報", "alarm", "錯誤", "error", "故障", "異常", "rtex", "int", "mot", "op", "過電流", "欠電壓", "保護"]):
        return "alarm"

    if any(k in q for k in ["維護", "保養", "潤滑", "拆卸", "接線", "硬體", "電路", "操作面板", "io", "ethercat版本"]):
        return "maintenance"

    if any(k in q for k in ["g碼", "m碼", "程式", "指令", "插補", "插值", "循環", "巨集", "macro", "刀補", "補正", "補償", "進給", "座標"]):
        return "programming"

    codes = extract_codes_from_query(query)
    if codes:
        # G/M 類通常是程式；INT/MOT/OP/RTEX/數字-字母通常是警報
        for c in codes:
            cn = normalize_for_search(c)
            if cn.startswith("g") or cn.startswith("m"):
                return "programming"
            if cn.startswith(("int", "mot", "op", "rtex")) or "-" in cn:
                return "alarm"

    return "general"


def score_payload(payload: Dict[str, Any], query: str, variants: List[str]) -> int:
    manual_type = str(payload.get("manual_type", "")).lower()
    content_type = str(payload.get("content_type", "")).lower()

    title = str(payload.get("title", ""))
    section = str(payload.get("section", ""))
    major_title = str(payload.get("major_title", ""))
    minor_title = str(payload.get("minor_title", ""))
    text = str(payload.get("text", ""))
    embedding_text = str(payload.get("embedding_text", ""))
    codes = [str(c) for c in payload.get("codes", [])]

    title_block = " ".join([title, section, major_title, minor_title])
    title_norm = normalize_for_search(title_block)
    text_head_norm = normalize_for_search(text[:1200])
    all_norm = normalize_for_search(" ".join([title_block, text, embedding_text, " ".join(codes)]))

    score = 0
    intent = detect_query_intent(query)

    if intent == "programming":
        if manual_type == "programming":
            score += 80
        elif manual_type == "maintenance":
            score += 15
        elif manual_type == "parameter_alarm":
            score += 5
    elif intent == "alarm":
        if manual_type == "parameter_alarm":
            score += 80
        elif manual_type == "maintenance":
            score += 35
        elif manual_type == "programming":
            score += 10
    elif intent == "maintenance":
        if manual_type == "maintenance":
            score += 80
        elif manual_type == "parameter_alarm":
            score += 25
        elif manual_type == "programming":
            score += 10
    else:
        # 不明確時平均，不硬壓特定手冊
        score += 10

    if content_type == "instruction":
        score += 15
    if content_type in ["alarm", "troubleshooting"] and intent == "alarm":
        score += 20
    if content_type in ["maintenance_instruction"] and intent == "maintenance":
        score += 20

    norm_variants = [normalize_for_search(v) for v in variants if v]

    for v in norm_variants:
        if not v:
            continue
        # 代碼在 codes 欄位完整命中，加最高分
        if any(v == normalize_for_search(c) for c in codes):
            score += 120
        if v in title_norm:
            score += 90
        if v in text_head_norm:
            score += 35
        if v in all_norm:
            score += 10

    # 有圖的資料稍微加分，讓同一章節中帶圖頁面更容易被帶回前端
    if payload.get("images"):
        score += 8

    return score


def keyword_search(query: str, limit: int = 5):
    """
    不用 sentence-transformers，適合 Render 小記憶體。
    流程：
    1. scroll 掃 Qdrant payload
    2. Python 做關鍵字命中
    3. 依問題類型排序
    4. 回傳前 limit 筆
    """
    matched_results = []
    offset = None
    variants = build_query_variants(query)

    if not variants:
        return []

    scanned = 0

    for _ in range(200):
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        scanned += len(points)

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
                if not v:
                    continue
                if v in search_text_raw or normalize_for_search(v) in search_text_compact:
                    matched = True
                    break

            if matched:
                payload["_score"] = score_payload(payload, query, variants)
                matched_results.append(payload)

        if offset is None:
            break

    matched_results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # 去重：避免同頁重複
    final_results = []
    seen = set()
    for item in matched_results:
        key = (
            item.get("source_pdf", ""),
            item.get("page", ""),
            item.get("section", ""),
            item.get("title", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        final_results.append(item)
        if len(final_results) >= limit:
            break

    return final_results


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
                "answer": f"查無相關資料。請確認 Qdrant collection：{COLLECTION_NAME} 是否已有對應 JSON 寫入。",
                "results": [],
                "images": [],
            }

        images = []
        for payload in results:
            for img in payload.get("images", []):
                if img and img not in images:
                    images.append(img)

        try:
            answer = generate_answer(query, results)
        except Exception as gemini_error:
            answer = (
                f"Gemini 生成失敗：{gemini_error}\n\n"
                f"以下是檢索到的原始資料：\n\n{build_context(results)[:3000]}"
            )

        # 不把 _score 顯示給前端
        clean_results = []
        for r in results:
            rr = dict(r)
            rr.pop("_score", None)
            clean_results.append(rr)

        return {
            "query": query,
            "collection": COLLECTION_NAME,
            "count": len(clean_results),
            "answer": answer,
            "results": clean_results,
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
