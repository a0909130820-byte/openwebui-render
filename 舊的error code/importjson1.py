import json
import time
import uuid
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType


# ===============================
# Qdrant Cloud 設定
# ===============================
QDRANT_URL = "https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODM2Mzk4MDUtYTVmNS00MzUyLWE2NWEtZWNlMWUxNWYxZTE3In0.SStK2mFTKzbEvbWc2r8B2s7TiXE68ETTKrPvmrkiJ7A"


def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ===============================
# 建立用來 Embedding 的完整文字
# ===============================
def build_embedding_text(row: Dict) -> str:
    error_code = str(row.get("error_code", "")).strip()
    error_message = str(row.get("error_message", "")).strip()
    cause_of_error = str(row.get("cause_of_error", "")).strip()
    error_correction = str(row.get("error_correction", "")).strip()
    text = str(row.get("text", "")).strip()

    # 如果 text 裡面沒有錯誤碼，自動補進去
    if error_code and error_code not in text:
        text = f"{error_code} {text}"

    full_text = f"""錯誤代碼：{error_code}
錯誤訊息：{error_message}
錯誤原因：{cause_of_error}
處理方式：{error_correction}
全文：{text}"""

    return full_text.strip()


# ===============================
# 產生穩定 Point ID
# ===============================
def make_point_id(row: Dict, embedding_text: str) -> str:
    """
    不使用 row['id']，避免不同 JSON 檔的 id 從 0 開始而互相覆蓋。
    這裡改用 UUID5：
    - 同一筆資料重跑：會更新同一個 point，不會重複塞爆
    - 新資料 / 新頁面 / 新文字：會新增到 Qdrant
    """
    source = str(row.get("source", "")).strip()
    page = str(row.get("page", "")).strip()
    error_code = str(row.get("error_code", "")).strip()
    key = f"{source}|{page}|{error_code}|{embedding_text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ===============================
# 一、Embedding
# ===============================
def embed_texts(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 50
) -> np.ndarray:

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


# ===============================
# 檢查 / 建立 Collection
# ===============================
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Collection 不存在，已建立：{collection_name}")
    else:
        info = client.get_collection(collection_name=collection_name)

        # 一般單向量 collection：info.config.params.vectors.size
        try:
            old_size = int(info.config.params.vectors.size)
        except Exception:
            old_size = None

        if old_size is not None and old_size != vector_size:
            raise RuntimeError(
                f"Collection 向量維度不一致：Qdrant 目前是 {old_size}，"
                f"但現在模型產生的是 {vector_size}。\n"
                f"請確認是否換了 embedding model，或改用新的 collection 名稱。"
            )

        print(f"✅ Collection 已存在，不刪除，直接追加資料：{collection_name}")


def ensure_payload_indexes(client: QdrantClient, collection_name: str):
    """建立常用 payload index；已存在時會略過錯誤。"""
    index_fields = {
        "error_code": PayloadSchemaType.KEYWORD,
        "source": PayloadSchemaType.KEYWORD,
        "page": PayloadSchemaType.INTEGER,
    }

    for field_name, field_schema in index_fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ 已建立 payload index：{field_name}")
        except Exception:
            print(f"ℹ️ payload index 可能已存在，略過：{field_name}")


# ===============================
# 二、寫入 Qdrant Cloud（追加，不刪除）
# ===============================
def write_to_qdrant_cloud(
    corpus: List[Dict],
    emb: np.ndarray,
    embedding_texts: List[str],
    qdrant_collection: str = "error_codes",
    upload_batch_size: int = 50
):

    if len(corpus) != emb.shape[0]:
        raise RuntimeError(
            f"資料筆數 ({len(corpus)}) 與向量數量 ({emb.shape[0]}) 不一致。"
        )

    if len(embedding_texts) != len(corpus):
        raise RuntimeError(
            f"embedding_texts 筆數 ({len(embedding_texts)}) 與 corpus 筆數 ({len(corpus)}) 不一致。"
        )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False
    )

    print("✅ 已連線到 Qdrant Cloud")
    print(client.get_collections())

    # 重要：不再 delete_collection()
    ensure_collection(
        client=client,
        collection_name=qdrant_collection,
        vector_size=int(emb.shape[1]),
    )

    ensure_payload_indexes(client, qdrant_collection)

    points = []

    for i, row in enumerate(corpus):
        error_code = str(row.get("error_code", "")).strip()
        error_message = str(row.get("error_message", "")).strip()
        cause_of_error = str(row.get("cause_of_error", "")).strip()
        error_correction = str(row.get("error_correction", "")).strip()
        source = str(row.get("source", "")).strip()

        try:
            page = int(row.get("page", 0))
        except Exception:
            page = 0

        text = str(row.get("text", "")).strip()

        if error_code and error_code not in text:
            text = f"{error_code} {text}"

        point_id = make_point_id(row, embedding_texts[i])

        points.append(
            PointStruct(
                id=point_id,
                vector=emb[i].tolist(),
                payload={
                    "text": text,
                    "embedding_text": embedding_texts[i],
                    "page": page,
                    "error_code": error_code,
                    "error_message": error_message,
                    "cause_of_error": cause_of_error,
                    "error_correction": error_correction,
                    "source": source,
                    "original_id": row.get("id", i),
                },
            )
        )

    total_points = len(points)
    sent = 0

    print(f"→ 準備追加寫入 Qdrant Cloud，共 {total_points} 筆")

    for batch_idx, batch in enumerate(chunk_list(points, upload_batch_size), start=1):
        client.upsert(
            collection_name=qdrant_collection,
            points=batch,
        )

        sent += len(batch)
        print(f"   - 已上傳批次 {batch_idx}，累計 {sent}/{total_points}")

    print(f"✅ 已追加寫入 Qdrant Cloud collection：{qdrant_collection}")


# ===============================
# 三、整合流程
# ===============================
def embed_and_store(
    corpus: List[Dict],
    model_name: str = "BAAI/bge-m3",
    embed_batch_size: int = 50,
    qdrant_collection: str = "error_codes",
    upload_batch_size: int = 50
):

    start = time.time()

    cleaned_corpus = []

    for row in corpus:
        error_code = str(row.get("error_code", "")).strip()
        text = str(row.get("text", "")).strip()
        error_message = str(row.get("error_message", "")).strip()
        cause_of_error = str(row.get("cause_of_error", "")).strip()
        error_correction = str(row.get("error_correction", "")).strip()

        # 只要其中一個欄位有內容就保留
        if error_code or text or error_message or cause_of_error or error_correction:
            cleaned_corpus.append(row)

    corpus = cleaned_corpus

    if not corpus:
        raise RuntimeError("corpus 裡沒有可用資料。")

    embedding_texts = [build_embedding_text(row) for row in corpus]

    print(f"[1/2] 產生 Embedding：{model_name}")
    print(f"共 {len(embedding_texts)} 筆資料")

    emb = embed_texts(
        embedding_texts,
        model_name=model_name,
        batch_size=embed_batch_size
    )

    print(f"[2/2] 追加寫入 Qdrant Cloud：{qdrant_collection}")

    write_to_qdrant_cloud(
        corpus=corpus,
        emb=emb,
        embedding_texts=embedding_texts,
        qdrant_collection=qdrant_collection,
        upload_batch_size=upload_batch_size
    )

    end = time.time()
    print(f"✅ 全部完成，花費時間：{end - start:.2f} 秒")


# ===============================
# 主程式
# ===============================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="chunk 後的 JSON 檔案")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--collection", default="error_codes")

    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    embed_and_store(
        corpus=corpus,
        model_name=args.model,
        qdrant_collection=args.collection
    )
