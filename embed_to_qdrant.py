import argparse
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "l2100_manuals"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/l2100_manuals.json"
    )

    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME
    )

    args = parser.parse_args()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise RuntimeError("缺少 QDRANT_URL")

    if not qdrant_api_key:
        raise RuntimeError("缺少 QDRANT_API_KEY")

    print("載入 embedding 模型...")
    model = SentenceTransformer(EMBED_MODEL)

    print("連接 Qdrant...")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=False,
        https=True,
        check_compatibility=False
    )

    print("測試 Qdrant 連線...")
    print(client.get_collections())

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    vector_size = model.get_sentence_embedding_dimension()

    print(f"Vector size: {vector_size}")

    try:
        if client.collection_exists(args.collection):
            print(f"刪除舊 collection：{args.collection}")
            client.delete_collection(args.collection)
    except Exception as e:
        print("collection_exists 失敗：", e)

    print(f"建立 collection：{args.collection}")

    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []

    print("開始 embedding...")

    for r in records:

        embedding_text = f"""
來源檔案:
{r.get("source_file", "")}

手冊類型:
{r.get("manual_type", "")}

標題:
{r.get("title", "")}

代碼:
{' '.join(r.get("codes", []))}

內容:
{r.get("text", "")}
"""

        vector = model.encode(
            embedding_text,
            normalize_embeddings=True
        ).tolist()

        payload = {
            "id": r.get("id"),
            "source_file": r.get("source_file"),
            "manual_type": r.get("manual_type"),
            "page": r.get("page"),
            "title": r.get("title"),
            "codes": r.get("codes", []),
            "text": r.get("text"),
            "images": r.get("images", [])
        }

        point = PointStruct(
            id=int(r["id"]),
            vector=vector,
            payload=payload
        )

        points.append(point)

    print(f"共 {len(points)} 筆")

    batch_size = 32

    for i in range(0, len(points), batch_size):

        batch = points[i:i + batch_size]

        client.upsert(
            collection_name=args.collection,
            points=batch
        )

        print(f"已上傳 {i + len(batch)}/{len(points)}")

    print("\n=== 完成 ===")
    print(f"Collection：{args.collection}")


if __name__ == "__main__":
    main()