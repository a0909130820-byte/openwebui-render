import gradio as gr

# 這裡要改成你的主程式檔名
# 例如你的查詢程式叫 query_with_ollama_new.py，就用下面這行
from query_with_ollama_new import (
    QDRANT_URL,
    get_embedder,
    retrieve_hits,
    build_context,
    generate_with_ollama,
)

# ===============================
# 跟你 VS Code 終端機版本保持一致的設定
# ===============================
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
QDRANT_COLLECTION = "error_codes"
OLLAMA_MODEL = "gemma3:12b"

TOP_K = 30
BOOST = 0.2
RERANK_TOPN = 10
MAX_CONTEXT_CHARS = 5000

# 啟動 UI 時先載入模型，避免每問一次重載一次
embedder = get_embedder(EMB_MODEL)


def ask_cnc_error(message, history):
    if not message.strip():
        return "請先輸入錯誤代碼或問題。"

    try:
        # 這段就是呼叫你原本 VS Code 裡的 RAG 查詢流程
        picked = retrieve_hits(
            question=message.strip(),
            emb_model_name=EMB_MODEL,
            embedder=embedder,
            qdrant_url=QDRANT_URL,
            qdrant_collection=QDRANT_COLLECTION,
            top_k=TOP_K,
            boost=BOOST,
            rerank_model=RERANK_MODEL,
            rerank_topn=RERANK_TOPN,
        )

        context = build_context(
            picked,
            text_col="text",
            max_chars=MAX_CONTEXT_CHARS
        )

        answer = generate_with_ollama(
            question=message.strip(),
            context=context,
            model_name=OLLAMA_MODEL
        )

        return answer

    except Exception as e:
        return f"系統錯誤：{e}"


demo = gr.ChatInterface(
    fn=ask_cnc_error,
    title="CNC 錯誤代碼查詢系統",
    description="輸入錯誤代碼或問題，系統會使用你原本的 Qdrant + Rerank + Ollama 流程回答。",
    textbox=gr.Textbox(
        placeholder="例如：1A0-01B8 或 主軸錯誤怎麼處理？",
        scale=7
    ),
    examples=[
        "1A0-01B8",
        "120-0007",
        "安全控制停止的原因是什麼？"
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )
