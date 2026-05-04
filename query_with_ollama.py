import re
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
    import ollama
    HAVE_OLLAMA = True
except Exception:
    HAVE_OLLAMA = False


ERRCODE_RE = re.compile(r"[A-Za-z]*\d[\w-]*")


# ===============================================
# 一、查詢前處理
# ===============================================
def get_embedder(model_name: str):
    print(f"→ 載入 Embedding 模型：{model_name}")
    return SentenceTransformer(model_name)


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
# 二、Qdrant 搜尋
# ===============================================
def qdrant_search(qdrant_url: str, collection: str, qv: np.ndarray, top_k: int = 5):
    base_url = qdrant_url.rstrip("/")
    url = f"{base_url}/collections/{collection}/points/search"

    payload = {
        "vector": qv[0].tolist(),
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }

    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant search HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"Qdrant search response unexpected: {data}")

    hits = []
    for p in data["result"]:
        pl = p.get("payload", {}) or {}
        pl["_score"] = p.get("score", 0.0)
        hits.append(pl)
    return hits


def apply_errorcode_boost_hits(hits: list, q: str, boost: float = 0.05):
    m = ERRCODE_RE.search(q)
    if not m:
        return hits

    code = m.group(0)
    for h in hits:
        if code and code in str(h.get("text", "")):
            h["_score"] += boost

    return sorted(hits, key=lambda x: -x["_score"])


# ===============================================
# 三、重排 + 組 context
# ===============================================
def rerank_hits(question: str, hits: list, text_col: str, model_name: str, topn: int = 3):
    if not HAVE_CE:
        print("（未安裝 CrossEncoder；略過重排）")
        return hits[:topn]

    print(f"→ CrossEncoder 重排：{model_name}（保留前 {topn} 個）")
    ce = CrossEncoder(model_name)
    pairs = [(question, str(h.get(text_col, ""))) for h in hits]
    scores = ce.predict(pairs)
    order = np.argsort(-np.array(scores))

    reranked = [hits[i] for i in order[:max(1, topn)]]
    for i, idx in enumerate(order[:max(1, topn)]):
        reranked[i]["_rerank"] = float(scores[idx])

    return reranked


def build_context(hits, text_col: str = "text", max_chars: int = 3000) -> str:
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


# ===============================================
# 四、Ollama 生成答案
# ===============================================
def generate_with_ollama(question: str, context: str, model_name: str) -> str:
    prompt = f"""你是技術助理。僅根據「知識片段」回答；若片段沒有答案就說不知道。
用中文條列重點，並在內容中引用對應頁碼。

知識片段：
{context}

問題：{question}
請提供原因、排查步驟與解法（若適用）。"""

    resp = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"]


# ===============================================
# 五、一次查詢
# ===============================================
def run_once(
    emb_model_name: str,
    question: str,
    gen_model: str | None,
    qdrant_url: str,
    qdrant_collection: str,
    top_k: int = 5,
    boost: float = 0.05,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_topn: int = 3,
):
    embedder = get_embedder(emb_model_name)

    q0 = expand_if_error_code(question.strip())
    qv = embed_query(maybe_prefix_query(q0, emb_model_name), embedder, normalize=True)

    hits = qdrant_search(qdrant_url, qdrant_collection, qv, top_k=top_k)
    hits = apply_errorcode_boost_hits(hits, question, boost=boost)

    picked = rerank_hits(question, hits, text_col="text", model_name=rerank_model, topn=rerank_topn)

    print("\n=== 重排後最佳片段 ===")
    for i, h in enumerate(picked, start=1):
        tag = f"(rerank={h.get('_rerank', 0):.4f})" if "_rerank" in h else ""
        body = str(h.get("text", "")).strip().replace("\n", " ")
        print(f"\n[{i}] {tag} page={h.get('page', '')} score={h.get('_score', 0):.4f}")
        print(body[:600])

    if gen_model:
        if not HAVE_OLLAMA:
            raise RuntimeError("未安裝 ollama 套件，請先：pip install ollama")

        context = build_context(picked, text_col="text", max_chars=3000)

        print("\n=== 生成答案 ===\n")
        out = generate_with_ollama(question, context, gen_model)
        print(out)


# ===============================================
# 六、互動模式
# ===============================================
def interactive_loop(
    emb_model_name: str,
    gen_model: str | None,
    qdrant_url: str,
    qdrant_collection: str,
    prompt_label: str = "錯誤代碼/問題：",
    top_k: int = 5,
    boost: float = 0.05,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_topn: int = 3,
):
    embedder = get_embedder(emb_model_name)

    print("\n=== 查詢就緒（Qdrant + Ollama） ===")
    print(f"- 使用嵌入模型：{emb_model_name}")
    print(f"- Qdrant：{qdrant_url} / collection={qdrant_collection}")
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

        q0 = expand_if_error_code(q)
        qv = embed_query(maybe_prefix_query(q0, emb_model_name), embedder, normalize=True)

        hits = qdrant_search(qdrant_url, qdrant_collection, qv, top_k=top_k)
        hits = apply_errorcode_boost_hits(hits, q, boost=boost)

        picked = rerank_hits(q, hits, text_col="text", model_name=rerank_model, topn=rerank_topn)

        print("\n=== 重排後最佳片段 ===")
        for i, h in enumerate(picked, start=1):
            tag = f"(rerank={h.get('_rerank', 0):.4f})" if "_rerank" in h else ""
            body = str(h.get("text", "")).strip().replace("\n", " ")
            print(f"\n[{i}] {tag} page={h.get('page', '')} score={h.get('_score', 0):.4f}")
            print(body[:600])

        if gen_model:
            if not HAVE_OLLAMA:
                print("\n（未安裝 ollama 套件，無法生成答案；請先安裝：pip install ollama）")
                continue

            context = build_context(picked, text_col="text", max_chars=3000)

            print("\n=== 生成答案 ===\n")
            try:
                out = generate_with_ollama(q, context, gen_model)
                print(out)
            except Exception as e:
                print(f"(生成失敗：{e})")


# ===============================================
# 七、main
# ===============================================
def main():
    ap = argparse.ArgumentParser(description="Qdrant 查詢 + CrossEncoder 重排 + Ollama 回答")
    ap.add_argument("--model", default="BAAI/bge-m3", help="Embedding 模型名稱")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333", help="Qdrant URL")
    ap.add_argument("--qdrant-collection", default="error_codes", help="Qdrant collection 名稱")
    ap.add_argument("--question", "-q", default="", help="直接發問；留空則進入互動模式")
    ap.add_argument("--gen-model", default="gemma3:12b", help="Ollama 模型名稱")
    ap.add_argument("--top-k", type=int, default=5, help="初步檢索片段數")
    ap.add_argument("--boost", type=float, default=0.05, help="錯誤代碼命中加權值")
    ap.add_argument("--prompt", default="錯誤代碼/問題：", help="互動模式提示文字")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3", help="CrossEncoder 模型名稱")
    ap.add_argument("--rerank-topn", type=int, default=3, help="重排後保留片段數")

    args = ap.parse_args()

    if args.question:
        run_once(
            emb_model_name=args.model,
            question=args.question.strip(),
            gen_model=args.gen_model.strip() or None,
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