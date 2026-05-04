import streamlit as st
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

# =========================
# 基本設定
# =========================
QDRANT_URL = "https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MjQ4NmMzOWMtNTRmNS00N2ExLWI4OTQtMjk0ZGVlNDZhYmJjIn0.BRYfqaufoZiCvnHNJuuSKe4Zo-Nzdx8j2jo-1VSolfE"
COLLECTION_NAME = "error_codes"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

# =========================
# UI
# =========================
st.set_page_config(page_title="CNC 錯誤代碼查詢系統", layout="wide")
st.title("CNC 錯誤代碼查詢系統")

# =========================
# 載入模型
# =========================
@st.cache_resource
def load_models():
    embed = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return embed, reranker

@st.cache_resource
def load_qdrant():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

embed_model, reranker = load_models()
client = load_qdrant()

# =========================
# Qdrant 查詢
# =========================
def search_qdrant(query, top_k=10):
    q_vec = embed_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0].tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k,
        with_payload=True
    )

    return result.points

# =========================
# 取得 payload 文字
# =========================
def get_text(payload):
    for key in ["text", "content", "page_content", "chunk"]:
        if key in payload and payload[key]:
            return str(payload[key])
    return str(payload)

# =========================
# Rerank
# =========================
def rerank(query, points, top_n=5):
    docs = []

    for p in points:
        payload = p.payload or {}
        text = get_text(payload)

        docs.append({
            "text": text,
            "payload": payload,
            "page": payload.get("page", "未知"),
            "score": p.score
        })

    if not docs:
        return []

    pairs = [[query, d["text"]] for d in docs]
    scores = reranker.predict(pairs)

    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

    return docs[:top_n]

# =========================
# Ollama 生成回答
# =========================
def ask_ollama(query, context):
    prompt = f"""
你是一個 CNC 錯誤代碼售服輔助系統。

請根據下方資料回答問題。
只能使用資料中的內容，不要自行補充或猜測。
請用繁體中文回答。

【資料】
{context}

【問題】
{query}

請整理成以下格式：

錯誤代碼：
錯誤說明：
原因：
處理方式：
頁碼：
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                }
            },
            timeout=120
        )

        st.write("🔍 Ollama status:", res.status_code)

        data = res.json()
        st.write("🔍 Ollama raw:", data)

        if res.status_code != 200:
            return f"Ollama 錯誤：{data}"

        answer = data.get("response", "").strip()

        if not answer:
            return "Ollama沒有產生文字"

        return answer

    except Exception as e:
        return f"Ollama錯誤：{e}"

# =========================
# UI 控制區
# =========================
query = st.text_input("輸入錯誤代碼或問題", placeholder="例如：120-0006")

top_k = st.slider("Qdrant 搜尋數量", 3, 20, 10)
top_n = st.slider("送給 Ollama 的資料數量", 1, 10, 3)

debug = st.checkbox("顯示檢索片段 Debug", True)

# =========================
# 查詢按鈕
# =========================
if st.button("查詢"):

    if not query.strip():
        st.warning("請先輸入錯誤代碼或問題")
    else:
        with st.spinner("正在查詢 Qdrant..."):
            points = search_qdrant(query, top_k)

        if not points:
            st.error("Qdrant 沒有查到資料")
        else:
            with st.spinner("正在重新排序 Rerank..."):
                docs = rerank(query, points, top_n)

            context = "\n\n---\n\n".join([
                f"頁碼：{d['page']}\n{d['text']}"
                for d in docs
            ])

            with st.spinner("正在呼叫 Ollama..."):
                answer = ask_ollama(query, context)

            st.subheader("回答")

            if "Ollama沒有產生文字" in answer or "Ollama錯誤" in answer:
                st.warning("Ollama 沒有正常產生回答，先直接顯示檢索結果")
                st.write(context)
            else:
                st.write(answer)

            if debug:
                st.subheader("檢索片段 Debug")

                for i, d in enumerate(docs, 1):
                    st.markdown(f"### 片段 {i}")
                    st.write("Qdrant 分數：", d["score"])
                    st.write("Rerank 分數：", d["rerank_score"])
                    st.write("頁碼：", d["page"])
                    st.text_area(
                        f"內容 {i}",
                        d["text"],
                        height=180,
                        key=f"debug_{i}"
                    )