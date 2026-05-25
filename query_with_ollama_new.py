import re
import os
import argparse
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
    HAVE_CE = True
except Exception:
    HAVE_CE = False

try:
    from google import genai
    HAVE_GEMINI = True
except Exception:
    HAVE_GEMINI = False


# ===============================
# Qdrant Cloud 設定
# ===============================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

ERRCODE_RE = re.compile(r"[A-Za-z0-9]+-[A-Za-z0-9]+")

_embedder = None
_cross_encoder = None
_embedder_model_name = None
_cross_encoder_model_name = None


def get_embedder(model_name: str):
    global _embedder, _embedder_model_name
    if _embedder is None or _embedder_model_name != model_name:
        print(f"→ 載入 Embedding 模型：{model_name}")
        _embedder = SentenceTransformer(model_name)
        _embedder_model_name = model_name
    return _embedder


def get_cross_encoder(model_name: str):
    global _cross_encoder, _cross_encoder_model_name
    if not HAVE_CE:
        return None
    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        print(f"→ 載入 CrossEncoder 模型：{model_name}")
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name
    return _cross_encoder


def expand_if_error_code(q: str) -> str:
    return f"{q} 的原因與處理方法是什麼？" if ERRCODE_RE.fullmatch(q.strip()) else q


def maybe_prefix_query(q: str, model_name: str) -> str:
    return ("query: " + q) if "bge" in model_name.lower() else q


def embed_query(q: str, model, normalize=True):
    v = model.encode([q], normalize_embeddings=normalize, convert_to_numpy=True)
    if v.dtype != np.float32:
        v = v.astype("float32", copy=False)
    return v


# ===============================================
# Qdrant 語意搜尋
# ===============================================
def qdrant_search(qdrant_url: str, collection: str, qv: np.ndarray, top_k: int = 30):
    base_url = qdrant_url.rstrip("/")
    url = f"{base_url}/collections/{collection}/points/search"

    payload = {
        "vector": qv[0].tolist(),
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }

    headers = {"api-key": QDRANT_API_KEY}

    resp = requests.post(url, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant search HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    hits = []
    for p in data.get("result", []):
        pl = p.get("payload", {}) or {}
        pl["_score"] = p.get("score", 0.0)
        hits.append(pl)

    return hits


# ===============================================
# Qdrant 錯誤代碼精準搜尋
# ===============================================
def qdrant_exact_code_search(qdrant_url: str, collection: str, code: str, limit: int = 10):
    base_url = qdrant_url.rstrip("/")
    url = f"{base_url}/collections/{collection}/points/scroll"

    payload = {
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [
                {
                    "key": "error_code",
                    "match": {
                        "value": code
                    }
                }
            ]
        }
    }

    headers = {"api-key": QDRANT_API_KEY}

    resp = requests.post(url, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant exact search HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    hits = []
    for p in data.get("result", {}).get("points", []):
        pl = p.get("payload", {}) or {}
        pl["_score"] = 999.0
        pl["_retrieval"] = "exact_error_code"
        hits.append(pl)

    return hits


def apply_errorcode_boost_hits(hits: list, q: str, boost: float = 0.2):
    m = ERRCODE_RE.search(q)
    if not m:
        return hits

    code = m.group(0)
    for h in hits:
        if code and code in str(h.get("text", "")):
            h["_score"] += boost

    return sorted(hits, key=lambda x: -x["_score"])


def rerank_hits(question: str, hits: list, text_col: str, model_name: str, topn: int = 10):
    ce = get_cross_encoder(model_name)
    if ce is None:
        print("（未安裝 CrossEncoder；略過重排）")
        return hits[:topn]

    print(f"→ CrossEncoder 重排（保留前 {topn} 個）")
    pairs = [(question, str(h.get(text_col, ""))) for h in hits]
    scores = ce.predict(pairs)
    order = np.argsort(-np.array(scores))

    reranked = [hits[i] for i in order[:max(1, topn)]]
    for i, idx in enumerate(order[:max(1, topn)]):
        reranked[i]["_rerank"] = float(scores[idx])

    return reranked


def retrieve_hits(
    question: str,
    emb_model_name: str,
    embedder,
    qdrant_url: str,
    qdrant_collection: str,
    top_k: int,
    boost: float,
    rerank_model: str,
    rerank_topn: int,
):
    question = question.strip()
    code_match = ERRCODE_RE.fullmatch(question)

    if code_match:
        code = question.upper()
        print(f"→ 先用 error_code 精準查詢：{code}")

        try:
            exact_hits = qdrant_exact_code_search(
                qdrant_url=qdrant_url,
                collection=qdrant_collection,
                code=code,
                limit=max(20, rerank_topn)
            )
        except Exception as e:
            print(f"→ error_code 精準查詢失敗：{e}")
            exact_hits = []

        if exact_hits:
            print(f"→ error_code 精準命中：{code}")
            return exact_hits[:rerank_topn]

        print("→ 精準查詢找不到，改用語意搜尋")

    q0 = expand_if_error_code(question)
    qv = embed_query(maybe_prefix_query(q0, emb_model_name), embedder, normalize=True)

    hits = qdrant_search(
        qdrant_url,
        qdrant_collection,
        qv,
        top_k=max(top_k, 50)
    )

    hits = apply_errorcode_boost_hits(hits, question, boost=boost)

    picked = rerank_hits(
        question,
        hits,
        text_col="text",
        model_name=rerank_model,
        topn=rerank_topn
    )

    return picked


def build_context(hits, max_chars: int = 7000) -> str:
    parts = []
    total = 0

    for h in hits:
        page = h.get("page", "?")
        section = h.get("section", "")
        major_title = h.get("major_title", "")
        minor_title = h.get("minor_title", "")
        content_type = h.get("content_type", "")
        table_columns = h.get("table_columns", [])
        version = h.get("version", "")
        date = h.get("date", "")
        maintenance_person = h.get("maintenance_person", "")
        text = str(h.get("text", "")).strip()

        block = f"""
page: {page}
section: {section}
major_title: {major_title}
minor_title: {minor_title}
content_type: {content_type}
"""

        if table_columns:
            block += f"table_columns: {table_columns}\n"

        if version:
            block += f"version: {version}\n"

        if date:
            block += f"date: {date}\n"

        if maintenance_person:
            block += f"maintenance_person: {maintenance_person}\n"

        block += f"text: {text}\n"

        block = block.strip()

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def generate_with_gemini(question: str, context: str, model_name: str, api_key: str) -> str:
    if not HAVE_GEMINI:
        raise RuntimeError("未安裝 Gemini 套件，請先執行：pip install google-genai==1.66.0")

    if not api_key:
        raise RuntimeError("找不到 GEMINI_API_KEY，請先設定環境變數。")

    client = genai.Client(api_key=api_key)

    prompt = f"""你是 L2100 CNC 車床維護與故障分析 AI 助理。

你只能根據「知識片段」內容回答。
禁止自行推測。
禁止補充知識片段不存在的內容。

========================
【知識片段 metadata 規則】
========================
知識片段可能包含以下 metadata：

- source_pdf：來源 PDF
- page：頁碼
- section：章節名稱
- major_title：大標題
- minor_title：小標題
- content_type：內容類型
    - text
    - table
    - revision_record
- table_columns：表格欄位名稱
- version：版本號
- date：日期
- maintenance_person：維修人員
- text：真正內容

========================
【回答規則】
========================
1. 只能使用知識片段中的內容回答。
2. 如果知識片段沒有提供答案，請回答：
   「資料片段未提供」。
3. 不可自行補充機械原理、推測故障原因。
4. 必須引用頁碼 page。
5. 如果有章節(section / major_title / minor_title)，要一起顯示。
6. 如果內容類型是 table：
   - 優先整理表格內容
   - 必須保留欄位意義
7. 如果內容類型是 revision_record：
   - 必須顯示版本號
   - 必須顯示日期
   - 必須顯示維修人員
8. 不要輸出無關內容。
9. 不要重複相同內容。
10. 如果問題是查詢修改紀錄：
   - 必須整理 revision_record
   - 按版本號列出
11. 如果問題是查詢功能、參數、IO、警報：
   - 必須優先顯示對應章節
   - 必須顯示頁碼
12. 如果知識片段有 table_columns：
   - 回答時要依照欄位整理
13. 如果知識片段中有多個相關頁面：
   - 請合併整理
   - 不要重複
14. 回答時不要輸出 markdown code block。
15. 客戶問中文就用中文回答。
16. 客戶問英文就用英文回答。
17. 回答內容要偏工程維修手冊風格。
18. 不要輸出「根據知識片段」。
19. 不要輸出「以下是整理」。

========================
【知識片段】
========================
{context}

========================
【問題】
========================
{question}

========================
【回答格式】
========================
請根據內容自動選擇適合格式：

【一般內容】
頁碼：
章節：
標題：
內容：

【表格內容】
頁碼：
章節：
表格欄位：
內容：

【修改紀錄】
版本：
日期：
維修人員：
內容：
頁碼：

如果知識片段沒有提供某欄位：
請填「資料片段未提供」。
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text


def print_hits(picked):
    print("\n=== 最佳片段 ===")
    for i, h in enumerate(picked, start=1):
        tag = f"(rerank={h.get('_rerank', 0):.4f})" if "_rerank" in h else ""
        body = f"""
section={h.get('section', '')}
major_title={h.get('major_title', '')}
minor_title={h.get('minor_title', '')}
content_type={h.get('content_type', '')}
text={str(h.get('text', '')).strip().replace(chr(10), ' ')}
"""
        print(f"\n[{i}] {tag} page={h.get('page', '')} score={h.get('_score', 0):.4f}")
        print(body[:600])


def run_once(
    emb_model_name: str,
    question: str,
    gen_model: str | None,
    gemini_api_key: str,
    qdrant_url: str,
    qdrant_collection: str,
    top_k: int = 30,
    boost: float = 0.2,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_topn: int = 10,
):
    embedder = get_embedder(emb_model_name)

    picked = retrieve_hits(
        question=question,
        emb_model_name=emb_model_name,
        embedder=embedder,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        top_k=top_k,
        boost=boost,
        rerank_model=rerank_model,
        rerank_topn=rerank_topn,
    )

    print_hits(picked)

    if gen_model:
        context = build_context(picked, max_chars=7000)

        print("\n=== CONTEXT DEBUG ===")
        print(context)

        print("\n=== Gemini 生成答案 ===\n")
        out = generate_with_gemini(question, context, gen_model, gemini_api_key)
        print(out)


def interactive_loop(
    emb_model_name: str,
    gen_model: str | None,
    gemini_api_key: str,
    qdrant_url: str,
    qdrant_collection: str,
    prompt_label: str = "錯誤代碼/問題：",
    top_k: int = 30,
    boost: float = 0.2,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_topn: int = 10,
):
    embedder = get_embedder(emb_model_name)
    get_cross_encoder(rerank_model)

    print("\n=== 查詢就緒（Qdrant Cloud + Gemini） ===")
    print(f"- 使用嵌入模型：{emb_model_name}")
    print(f"- 生成模型：{gen_model}")
    print(f"- Qdrant Cloud：{qdrant_url} / collection={qdrant_collection}")
    print("提示：輸入空行離開。")

    while True:
        try:
            q = input(f"\n{prompt_label}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            print("Bye.")
            break

        picked = retrieve_hits(
            question=q,
            emb_model_name=emb_model_name,
            embedder=embedder,
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            top_k=top_k,
            boost=boost,
            rerank_model=rerank_model,
            rerank_topn=rerank_topn,
        )

        print_hits(picked)

        if gen_model:
            context = build_context(picked, max_chars=7000)

            print("\n=== CONTEXT DEBUG ===")
            print(context)

            print("\n=== Gemini 生成答案 ===\n")
            try:
                out = generate_with_gemini(q, context, gen_model, gemini_api_key)
                print(out)
            except Exception as e:
                print(f"(生成失敗：{e})")



# =====================================================
# FastAPI Web UI / API 包裝層
# 注意：下面只是把原本 RAG 函式包成 Web API，不改原本 CLI 邏輯。
# CLI 仍可用：python query_with_ollama_new.py
# Web 可用：uvicorn query_with_ollama_new:app --host 0.0.0.0 --port 10000
# =====================================================
app = FastAPI(title="CNC RAG Web UI")


class ChatRequest(BaseModel):
    question: str
    emb_model_name: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    qdrant_url: str = os.getenv("QDRANT_URL", QDRANT_URL)
    qdrant_collection: str = os.getenv("COLLECTION_NAME", "error_codes")
    gen_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    top_k: int = 30
    boost: float = 0.2
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_topn: int = 10


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CNC RAG Web UI</title>
  <style>
    body { font-family: Arial, 'Microsoft JhengHei', sans-serif; background:#f5f6f8; margin:0; }
    .wrap { max-width: 920px; margin: 40px auto; padding: 24px; background:white; border-radius:14px; box-shadow:0 8px 28px rgba(0,0,0,.08); }
    h1 { margin-top:0; font-size:26px; }
    textarea { width:100%; min-height:110px; font-size:16px; padding:12px; box-sizing:border-box; border:1px solid #ccc; border-radius:10px; resize:vertical; }
    button { margin-top:12px; padding:10px 18px; font-size:16px; border:0; border-radius:10px; cursor:pointer; background:#111827; color:white; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .box { margin-top:20px; padding:16px; border-radius:10px; background:#f3f4f6; white-space:pre-wrap; line-height:1.6; }
    .err { color:#b91c1c; }
    .small { color:#6b7280; font-size:13px; margin-top:8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CNC RAG 問答系統</h1>
    <textarea id="q" placeholder="輸入錯誤代碼或問題，例如：130-009A 的原因與處理方法？"></textarea>
    <br />
    <button id="btn" onclick="ask()">送出查詢</button>
    <div class="small">API：POST /chat</div>
    <div id="ans" class="box">等待輸入問題...</div>
  </div>

<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  const ans = document.getElementById('ans');
  const btn = document.getElementById('btn');
  if (!q) { ans.textContent = '請先輸入問題'; return; }
  btn.disabled = true;
  ans.className = 'box';
  ans.textContent = '查詢中，第一次載入模型會比較久...';
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
    ans.textContent = data.answer || '沒有回答內容';
  } catch (e) {
    ans.className = 'box err';
    ans.textContent = '錯誤：' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"ok": True, "message": "CNC RAG API running"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return {"question": req.question, "answer": "請輸入問題", "hits": []}

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    embedder = get_embedder(req.emb_model_name)

    picked = retrieve_hits(
        question=question,
        emb_model_name=req.emb_model_name,
        embedder=embedder,
        qdrant_url=req.qdrant_url,
        qdrant_collection=req.qdrant_collection,
        top_k=req.top_k,
        boost=req.boost,
        rerank_model=req.rerank_model,
        rerank_topn=req.rerank_topn,
    )

    context = build_context(picked, max_chars=7000)

    answer = generate_with_gemini(
        question,
        context,
        req.gen_model,
        gemini_api_key,
    )

    return {
        "question": question,
        "answer": answer,
        "hits": picked,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Qdrant Cloud 查詢 + 錯誤碼精準搜尋 + CrossEncoder 重排 + Gemini 回答"
    )

    ap.add_argument("--model", default="BAAI/bge-m3", help="Embedding 模型名稱")
    ap.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant Cloud URL")
    ap.add_argument("--qdrant-collection", default="error_codes", help="Qdrant collection 名稱")
    ap.add_argument("--question", "-q", default="", help="直接發問；留空則進入互動模式")
    ap.add_argument("--gen-model", default="gemini-2.5-flash", help="Gemini 模型名稱")
    ap.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API Key")
    ap.add_argument("--top-k", type=int, default=30, help="初步檢索片段數")
    ap.add_argument("--boost", type=float, default=0.2, help="錯誤代碼命中加權值")
    ap.add_argument("--prompt", default="錯誤代碼/問題：", help="互動模式提示文字")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3", help="CrossEncoder 模型名稱")
    ap.add_argument("--rerank-topn", type=int, default=10, help="重排後保留片段數")

    args = ap.parse_args()

    if args.question:
        run_once(
            emb_model_name=args.model,
            question=args.question.strip(),
            gen_model=args.gen_model.strip() or None,
            gemini_api_key=args.gemini_api_key,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            top_k=args.top_k,
            boost=args.boost,
            rerank_model=args.rerank_model,
            rerank_topn=args.rerank_topn,
        )
    else:
        interactive_loop(
            emb_model_name=args.model,
            gen_model=args.gen_model.strip() or None,
            gemini_api_key=args.gemini_api_key,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            prompt_label=args.prompt,
            top_k=args.top_k,
            boost=args.boost,
            rerank_model=args.rerank_model,
            rerank_topn=args.rerank_topn,
        )


if __name__ == "__main__":
    main()