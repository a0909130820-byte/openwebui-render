
import json
import time
import uuid
from typing import List, Dict, Any

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
# 讀取 JSON
# 支援兩種格式：
# 1. [ {...}, {...} ]
# 2. { "documents": [ {...}, {...} ] }
# ===============================
def load_json_documents(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "documents" in data:
        return data["documents"]

    raise RuntimeError("JSON 格式錯誤：必須是 list，或是包含 documents 的 dict。")


# ===============================
# 把你的 JSON 轉成 Qdrant 可寫入資料
# 你的 JSON metadata 格式：
# metadata.page
# metadata.source_pdf
# metadata.section
# metadata.major_title
# metadata.minor_title
# metadata.content_type
# metadata.table_columns
# metadata.revision_records
# ===============================
def flatten_documents(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for doc in corpus:
        metadata = doc.get("metadata", {}) or {}

        page = metadata.get("page", doc.get("page", 0))
        source_pdf = metadata.get("source_pdf", "")
        section = metadata.get("section", "")
        major_title = metadata.get("major_title", "")
        minor_title = metadata.get("minor_title", "")
        content_type = metadata.get("content_type", "")
        table_columns = metadata.get("table_columns", [])

        # 第 4~7 頁的 revision_table
        # 拆成每一筆版本紀錄，這樣 embedding 和搜尋會比較準
        if content_type == "revision_table":
            revision_records = metadata.get("revision_records", []) or []

            for idx, record in enumerate(revision_records):
                version = str(record.get("version", "")).strip()
                date = str(record.get("date", "")).strip()
                maintenance_person = str(record.get("maintenance_person", "")).strip()
                text = str(record.get("text", "")).strip()

                if not (version or date or maintenance_person or text):
                    continue

                rows.append({
                    "id": f"{doc.get('id', '')}_rev_{idx + 1}",
                    "page": page,
                    "source_pdf": source_pdf,
                    "section": section,
                    "major_title": major_title,
                    "minor_title": minor_title,
                    "content_type": "revision_record",
                    "table_columns": ["version", "date", "maintenance_person", "text"],
                    "version": version,
                    "date": date,
                    "maintenance_person": maintenance_person,
                    "text": text,
                    "original_metadata": metadata,
                })

            continue

        # 一般文字 / 表格頁面
        text = str(doc.get("text", "")).strip()

        # 如果沒有 text，就不要寫入
        if not text:
            continue

        rows.append({
            "id": doc.get("id", len(rows) + 1),
            "page": page,
            "source_pdf": source_pdf,
            "section": section,
            "major_title": major_title,
            "minor_title": minor_title,
            "content_type": content_type,
            "table_columns": table_columns,
            "version": "",
            "date": "",
            "maintenance_person": "",
            "text": text,
            "original_metadata": metadata,
        })

    return rows


# ===============================
# 建立用來 Embedding 的完整文字
# ===============================
def build_embedding_text(row: Dict[str, Any]) -> str:
    page = str(row.get("page", "")).strip()
    source_pdf = str(row.get("source_pdf", "")).strip()
    section = str(row.get("section", "")).strip()
    major_title = str(row.get("major_title", "")).strip()
    minor_title = str(row.get("minor_title", "")).strip()
    content_type = str(row.get("content_type", "")).strip()
    table_columns = row.get("table_columns", [])
    version = str(row.get("version", "")).strip()
    date = str(row.get("date", "")).strip()
    maintenance_person = str(row.get("maintenance_person", "")).strip()
    text = str(row.get("text", "")).strip()

    if isinstance(table_columns, list):
        table_columns_text = "、".join([str(x) for x in table_columns])
    else:
        table_columns_text = str(table_columns)

    parts = [
        f"來源檔案：{source_pdf}",
        f"頁碼：{page}",
        f"章節：{section}",
        f"大標題：{major_title}",
        f"小標題：{minor_title}",
        f"內容類型：{content_type}",
    ]

    if table_columns_text:
        parts.append(f"表格欄位：{table_columns_text}")

    if version or date or maintenance_person:
        parts.extend([
            f"版本：{version}",
            f"日期：{date}",
            f"維修人員：{maintenance_person}",
        ])

    parts.append(f"內容：{text}")

    return "\n".join([p for p in parts if p and not p.endswith("：")])


# ===============================
# 產生穩定 Point ID
# ===============================
def make_point_id(row: Dict[str, Any], embedding_text: str) -> str:
    source_pdf = str(row.get("source_pdf", "")).strip()
    page = str(row.get("page", "")).strip()
    content_type = str(row.get("content_type", "")).strip()
    row_id = str(row.get("id", "")).strip()
    version = str(row.get("version", "")).strip()

    key = f"{source_pdf}|{page}|{content_type}|{row_id}|{version}|{embedding_text}"
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
    """
    建立常用 payload index。
    Qdrant 已存在 index 時會略過。
    """
    index_fields = {
        "source_pdf": PayloadSchemaType.KEYWORD,
        "section": PayloadSchemaType.KEYWORD,
        "major_title": PayloadSchemaType.KEYWORD,
        "minor_title": PayloadSchemaType.KEYWORD,
        "content_type": PayloadSchemaType.KEYWORD,
        "version": PayloadSchemaType.KEYWORD,
        "date": PayloadSchemaType.KEYWORD,
        "maintenance_person": PayloadSchemaType.KEYWORD,
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
    rows: List[Dict[str, Any]],
    emb: np.ndarray,
    embedding_texts: List[str],
    qdrant_collection: str = "maintenance_manual",
    upload_batch_size: int = 50
):

    if len(rows) != emb.shape[0]:
        raise RuntimeError(
            f"資料筆數 ({len(rows)}) 與向量數量 ({emb.shape[0]}) 不一致。"
        )

    if len(embedding_texts) != len(rows):
        raise RuntimeError(
            f"embedding_texts 筆數 ({len(embedding_texts)}) 與 rows 筆數 ({len(rows)}) 不一致。"
        )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False
    )

    print("✅ 已連線到 Qdrant Cloud")
    print(client.get_collections())

    ensure_collection(
        client=client,
        collection_name=qdrant_collection,
        vector_size=int(emb.shape[1]),
    )

    ensure_payload_indexes(client, qdrant_collection)

    points = []

    for i, row in enumerate(rows):
        try:
            page = int(row.get("page", 0))
        except Exception:
            page = 0

        table_columns = row.get("table_columns", [])
        if not isinstance(table_columns, list):
            table_columns = [str(table_columns)]

        point_id = make_point_id(row, embedding_texts[i])

        points.append(
            PointStruct(
                id=point_id,
                vector=emb[i].tolist(),
                payload={
                    "text": str(row.get("text", "")).strip(),
                    "embedding_text": embedding_texts[i],

                    # 你的新版 JSON metadata
                    "page": page,
                    "source_pdf": str(row.get("source_pdf", "")).strip(),
                    "section": str(row.get("section", "")).strip(),
                    "major_title": str(row.get("major_title", "")).strip(),
                    "minor_title": str(row.get("minor_title", "")).strip(),
                    "content_type": str(row.get("content_type", "")).strip(),
                    "table_columns": table_columns,

                    # 第 4~7 頁修改紀錄用
                    "version": str(row.get("version", "")).strip(),
                    "date": str(row.get("date", "")).strip(),
                    "maintenance_person": str(row.get("maintenance_person", "")).strip(),

                    # 保留原始資訊方便除錯
                    "original_id": row.get("id", i),
                    "original_metadata": row.get("original_metadata", {}),
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
    corpus: List[Dict[str, Any]],
    model_name: str = "BAAI/bge-m3",
    embed_batch_size: int = 50,
    qdrant_collection: str = "maintenance_manual",
    upload_batch_size: int = 50
):

    start = time.time()

    rows = flatten_documents(corpus)

    if not rows:
        raise RuntimeError("JSON 裡沒有可用資料。")

    embedding_texts = [build_embedding_text(row) for row in rows]

    print(f"[1/2] 產生 Embedding：{model_name}")
    print(f"原始 documents：{len(corpus)}")
    print(f"實際寫入 rows：{len(rows)}")

    emb = embed_texts(
        embedding_texts,
        model_name=model_name,
        batch_size=embed_batch_size
    )

    print(f"[2/2] 追加寫入 Qdrant Cloud：{qdrant_collection}")

    write_to_qdrant_cloud(
        rows=rows,
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

    ap.add_argument("--input", required=True, help="JSON 檔案")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--collection", default="maintenance_manual")

    args = ap.parse_args()

    corpus = load_json_documents(args.input)

    embed_and_store(
        corpus=corpus,
        model_name=args.model,
        qdrant_collection=args.collection
    )
