import re
import os
import argparse
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
    HAVE_CE = True
except Exception:
    HAVE_CE = False

try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except Exception:
    HAVE_GEMINI = False


# ===============================
# Qdrant Cloud 設定
# ===============================
QDRANT_URL = "https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODM2Mzk4MDUtYTVmNS00MzUyLWE2NWEtZWNlMWUxNWYxZTE3In0.SStK2mFTKzbEvbWc2r8B2s7TiXE68ETTKrPvmrkiJ7A"

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


def build_context(hits, text_col: str = "text", max_chars: int = 5000) -> str:
    parts, total = [], 0

    for h in hits:
        prefix = f"(page={h.get('page', '?')}) "
        t = prefix + str(h.get(text_col, "")).strip().replace("\n", " ")

        if total + len(t) > max_chars:
            t = t[:max(0, max_chars - total)]

        parts.append(t)
        total += len(t)

        if total >= max_chars:
            break

    return "\n\n".join(parts)


def generate_with_gemini(question: str, context: str, model_name: str, api_key: str) -> str:
    if not HAVE_GEMINI:
        raise RuntimeError("未安裝 Gemini 套件，請先執行：pip install google-generativeai")

    if not api_key:
        raise RuntimeError("找不到 GEMINI_API_KEY，請先設定環境變數。")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    prompt = f"""你是 CNC 超強助理維修工程師，必須解決客戶一切問題但你必須遵照以下規則。

回答規則：
1. 只能根據「知識片段」回答。
2. 不可以補充知識片段以外的內容。
3. 不可以自行推測知識片段沒有提供的內容。
4. 如果某個欄位在知識片段沒有寫，該欄位請填「資料片段未提供」。
5. 不可以輸出任何與問題無關的內容。
6. 必須引用頁碼。
7. 只能輸出下面四個項目，不要輸出其他說明文字。
8. 必須完整內容輸出。
9. Error message、Cause of error、Error correction 三個項目，不能有遺漏。
10. 如果知識片段中同一個頁面有多個相關內容，請合併在一起回答，不要分開成多個項目。
11. Error message 後面不顯示錯誤代碼，但要顯示錯誤訊息。
12. 客戶問中文或英文問題時不用另外補充語言說明，直接回答就好。
13. 客戶如果沒有問錯誤代碼，在回答時要告訴他錯誤代碼是多少、在第幾頁。

知識片段：
{context}

問題：{question}

請固定用以下格式回答：
1. 錯誤代碼:（如果知識片段有提供錯誤代碼就寫出來，沒有的話就寫「資料片段未提供」）
2. Error message：

3. Cause of error：

4. Error correction：
-
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
        }
    )

    return response.text


def print_hits(picked):
    print("\n=== 最佳片段 ===")
    for i, h in enumerate(picked, start=1):
        tag = f"(rerank={h.get('_rerank', 0):.4f})" if "_rerank" in h else ""
        body = str(h.get("text", "")).strip().replace("\n", " ")
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
        context = build_context(picked, text_col="text", max_chars=5000)

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
            context = build_context(picked, text_col="text", max_chars=5000)

            print("\n=== CONTEXT DEBUG ===")
            print(context)

            print("\n=== Gemini 生成答案 ===\n")
            try:
                out = generate_with_gemini(q, context, gen_model, gemini_api_key)
                print(out)
            except Exception as e:
                print(f"(生成失敗：{e})")


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