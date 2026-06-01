import os
import json
import re
import time
from typing import List
from urllib.parse import quote

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

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/page_images", StaticFiles(directory="page_images"), name="page_images")


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "l2100_manuals"
BASE_URL = "https://cnc-error-ai.onrender.com"


class QueryRequest(BaseModel):
    query: str
    manual_type: str = "all"
    use_ollama: bool = True


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{path} 載入失敗：{e}")
        return default


SECTION_MAP = load_json("section_map.json", [])


qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "L2100 Manual AI API running",
        "collection": COLLECTION_NAME,
        "section_count": len(SECTION_MAP)
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


def normalize(text):
    return str(text).upper().replace(" ", "").replace("　", "")


def extract_keywords(query: str):
    q = normalize(query)

    keys = re.findall(
        r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|ETHERCAT|參數\d+",
        q,
        flags=re.IGNORECASE
    )

    out = []

    for k in keys:
        k = normalize(k)
        if k not in out:
            out.append(k)

    if not out:
        out = [q]

    return out


def keyword_search(
    query: str,
    manual_type: str = "all",
    limit: int = 10
):
    keys = extract_keywords(query)

    results = []
    offset = None

    for _ in range(50):
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

                if payload.get("manual_type") != manual_type:
                    continue

            text = normalize(
                str(payload.get("title", "")) + " " +
                str(payload.get("text", "")) + " " +
                " ".join(payload.get("codes", []))
            )

            score = 0

            for key in keys:
                if key in text:
                    score += 1

            if score > 0:
                payload["_score"] = score
                results.append(payload)

        if offset is None:
            break

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    unique = []
    seen = set()

    for r in results:
        uid = (
            str(r.get("source_file", "")),
            str(r.get("page", "")),
            str(r.get("text", ""))[:100]
        )

        if uid not in seen:
            seen.add(uid)
            unique.append(r)

        if len(unique) >= limit:
            break

    return unique


def search_sections(query: str):
    keys = extract_keywords(query)
    matched = []

    for sec in SECTION_MAP:
        sec_text = normalize(
            str(sec.get("section", "")) + " " +
            " ".join(sec.get("codes", []))
        )

        score = 0

        for key in keys:
            if key in sec_text:
                score += 100

        if score > 0:
            matched.append((score, sec))

    matched.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in matched[:1]]


def get_section_page_images(section, results=None):
    images = []

    manual_type = section.get("manual_type", "")
    source_file = section.get("source_file", "")
    section_name = section.get("section", "")

    start_page = int(section.get("start_page", 0))
    end_page = int(section.get("end_page", start_page))

    max_page = end_page + 1
    pages = set()

    for page in range(start_page, end_page + 1):
        pages.add(page)

    if results:
        for r in results:
            if r.get("source_file") != source_file:
                continue

            try:
                page = int(r.get("page"))
            except:
                continue

            if start_page <= page <= max_page:
                pages.add(page)

    for page in sorted(pages):
        path = f"/page_images/{manual_type}/page_{page:04d}.png"

        images.append({
            "url": BASE_URL + quote(path, safe="/:"),
            "page": page,
            "source_file": source_file,
            "section": section_name
        })

    return images


def build_context(results: List[dict]):
    context = ""

    for i, r in enumerate(results, 1):
        context += f"""
【資料 {i}】

來源：
{r.get("source_file", "")}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", "")}

內容：
{r.get("text", "")[:3000]}
"""

    return context


def build_sources(results: List[dict]):
    sources = []
    seen = set()

    for r in results:
        source_file = r.get("source_file", "")
        page = r.get("page", "")

        key = (source_file, str(page))

        if source_file and page and key not in seen:
            seen.add(key)
            sources.append({
                "source_file": source_file,
                "page": page,
                "title": r.get("title", "")
            })

    return sources


def generate_answer(query: str, results: List[dict]):
    context = build_context(results)

    if not gemini_client:
        return context[:4000]

    prompt = f"""
你是 L2100 車床手冊 AI 助理。

只能根據以下手冊回答：
1. L2100 車床程式說明手冊
2. L2100 車床中文維護手冊
3. L2100 車床參數警報手冊

禁止幻想不存在資訊。

使用者問題：
{query}

手冊內容：
{context}

請使用繁體中文整理：
1. 查詢重點
2. 功能說明
3. 使用注意事項
4. 來源頁碼
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        print("Gemini 失敗：", e)

        fallback = "【Gemini 額度不足，改用手冊原文模式】\n\n"

        for i, r in enumerate(results, 1):
            fallback += f"""
========================
資料 {i}
========================

來源：
{r.get("source_file", "")}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", "")}

內容：
{r.get("text", "")[:2500]}

"""

        return fallback


@app.post("/search")
def search(req: QueryRequest):
    start_time = time.time()
    query = req.query.strip()

    try:
        results = keyword_search(
            query,
            manual_type=req.manual_type,
            limit=10
        )

        if not results:
            return {
                "query": query,
                "count": 0,
                "elapsed": round(time.time() - start_time, 2),
                "answer": "查無相關資料。",
                "results": [],
                "sources": [],
                "images": []
            }

        sections = search_sections(query)

        images = []

        if sections:
            images = get_section_page_images(sections[0], results)

        try:
            answer = generate_answer(query, results)
        except Exception as e:
            print("Gemini 失敗：", e)
            answer = build_context(results)

        elapsed = round(time.time() - start_time, 2)

        return {
            "query": query,
            "count": len(results),
            "elapsed": elapsed,
            "answer": answer,
            "results": results,
            "sources": build_sources(results),
            "images": images
        }

    except Exception as e:
        print("Search Error:", e)

        return {
            "query": query,
            "count": 0,
            "elapsed": round(time.time() - start_time, 2),
            "answer": f"後端錯誤：{e}",
            "results": [],
            "sources": [],
            "images": []
        }