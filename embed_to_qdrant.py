import json
import time
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def chunk_list(lst, batch_size):
    """把大的 list 切成小批次"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ===============================================
# 一、Embedding
# ===============================================
def embed_texts(
    texts,
    model_name="BAAI/bge-m3",
    batch_size=25
) -> np.ndarray:
    """把文字轉成向量"""
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if emb.dtype != np.float32:
        emb = emb.astype("float32", copy=False)
    return emb


# ===============================================
# 二、寫入 Qdrant
# ===============================================
def write_to_qdrant(
    corpus: List[Dict],
    emb: np.ndarray,
    qdrant_url: str = "http://127.0.0.1:6333",
    qdrant_api_key: str | None = None,
    qdrant_collection: str = "error_codes",
    upload_batch_size: int = 300
):
    """
    corpus 每筆資料格式建議：
    {
        "id": 0,
        "page": 1,
        "error_code": "101",
        "source": "錯誤清單",
        "text": "這是一段 chunk 文字"
    }
    """
    if len(corpus) != emb.shape[0]:
        raise RuntimeError(f"資料筆數 ({len(corpus)}) 與向量數量 ({emb.shape[0]}) 不一致。")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key or None
    )

    client.recreate_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(
            size=int(emb.shape[1]),
            distance=Distance.COSINE,
        ),
    )

    points = []
    for i, row in enumerate(corpus):
        points.append(
            PointStruct(
                id=int(row["id"]),
                vector=emb[i].tolist(),
                payload={
                    "text": row.get("text", ""),
                    "page": int(row.get("page", 0)),
                    "error_code": row.get("error_code", ""),
                    "source": row.get("source", ""),
                },
            )
        )

    total_points = len(points)
    sent = 0

    print(f"→ 準備寫入 Qdrant，共 {total_points} 筆，批次大小 = {upload_batch_size}")

    for batch_idx, batch in enumerate(chunk_list(points, upload_batch_size), start=1):
        client.upsert(
            collection_name=qdrant_collection,
            points=batch,
        )
        sent += len(batch)
        print(f"   - 已上傳批次 {batch_idx}，本批 {len(batch)} 筆，累計 {sent}/{total_points}")

    print(f"✅ 已寫入 Qdrant collection：{qdrant_collection}（{total_points} 筆）")


# ===============================================
# 三、整合：chunk資料 → embedding → qdrant
# ===============================================
def embed_and_store(
    corpus: List[Dict],
    model_name: str = "BAAI/bge-m3",
    embed_batch_size: int = 25,
    qdrant_url: str = "http://127.0.0.1:6333",
    qdrant_api_key: str | None = None,
    qdrant_collection: str = "error_codes",
    upload_batch_size: int = 300
):
    start = time.time()

    texts = [row["text"] for row in corpus if str(row.get("text", "")).strip()]
    if not texts:
        raise RuntimeError("corpus 裡沒有可用的 text。")

    corpus = [row for row in corpus if str(row.get("text", "")).strip()]

    print(f"[1/2] 產生 Embedding：{model_name}，共 {len(texts)} 筆")
    emb = embed_texts(texts, model_name=model_name, batch_size=embed_batch_size)

    print(f"[2/2] 寫入 Qdrant：{qdrant_collection}")
    write_to_qdrant(
        corpus=corpus,
        emb=emb,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        upload_batch_size=upload_batch_size
    )

    print(f"✅ 完成，耗時 {time.time() - start:.1f} 秒")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunk 後的 JSON")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="error_codes")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    embed_and_store(
        corpus=corpus,
        model_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection
    )