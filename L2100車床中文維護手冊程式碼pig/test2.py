
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


TEXT_COL_CANDIDATES = ["text", "content", "page_content", "chunk"]


def read_json_anyencoding(json_path: str, preferred_encoding: str | None = None):
    tries = [e for e in [preferred_encoding, "utf-8", "utf-8-sig", "cp950", "big5", "latin-1"] if e]

    with open(json_path, "rb") as f:
        raw = f.read()

    last_error = None

    for enc in tries:
        try:
            text = raw.decode(enc)
            print(f"讀取成功，使用編碼：{enc}")
            return json.loads(text)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"JSON 讀取或解析失敗：{last_error}")


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = (text or "").strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必須大於 0")

    if overlap < 0:
        raise ValueError("overlap 不能小於 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必須小於 chunk_size")

    # 中文標點、英文標點、換行都可切
    sents = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text)
    sents = [s.strip() for s in sents if s.strip()]

    if not sents:
        return [text]

    chunks = []
    buf = ""

    for s in sents:
        # 單句太長就硬切
        if len(s) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""

            step = chunk_size - overlap
            for i in range(0, len(s), step):
                piece = s[i:i + chunk_size].strip()
                if piece:
                    chunks.append(piece)

            continue

        if len(buf) + len(s) + 1 <= chunk_size:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)

            if overlap > 0 and len(buf) > overlap:
                buf = (buf[-overlap:] + " " + s).strip()
            else:
                buf = s

    if buf:
        chunks.append(buf)

    return chunks


def normalize_input_items(data: Any) -> List[Dict[str, Any]]:
    """
    支援：
    1. [ {...}, {...} ]
    2. { "documents": [ {...}, {...} ] }
    """
    if isinstance(data, dict):
        if "documents" in data and isinstance(data["documents"], list):
            return data["documents"]

        for key in ("items", "data", "records", "list"):
            if key in data and isinstance(data[key], list):
                return data[key]

        return [data]

    if isinstance(data, list):
        return data

    if isinstance(data, str):
        return [{"id": 0, "page": 1, "metadata": {}, "text": data}]

    return [{"id": 0, "page": 1, "metadata": {}, "text": str(data)}]


def get_text_from_item(item: Dict[str, Any]) -> str:
    for k in TEXT_COL_CANDIDATES:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k].strip()

    parts = []

    for k in ["Error message", "Cause of error", "Error correction"]:
        if k in item and isinstance(item[k], str) and item[k].strip():
            parts.append(f"{k}: {item[k].strip()}")

    return "\n".join(parts).strip()


def build_revision_chunks(
    item: Dict[str, Any],
    cid_start: int,
    chunk_size: int,
    overlap: int
) -> tuple[List[Dict[str, Any]], int]:
    """
    第 4～7 頁修改紀錄：
    metadata.revision_records 每一筆都獨立變成一個 chunk。
    不依賴外層 text，避免 text 為空被跳過。
    """
    chunks_out = []
    cid = cid_start

    metadata = item.get("metadata", {}) or {}
    revision_records = metadata.get("revision_records", []) or []

    page = metadata.get("page", item.get("page", ""))
    source_pdf = metadata.get("source_pdf", item.get("source", ""))
    section = metadata.get("section", "")
    major_title = metadata.get("major_title", "")
    minor_title = metadata.get("minor_title", "")
    content_type = metadata.get("content_type", "revision_table")

    for idx, record in enumerate(revision_records, start=1):
        version = str(record.get("version", "")).strip()
        date = str(record.get("date", "")).strip()
        maintenance_person = str(record.get("maintenance_person", "")).strip()
        text = str(record.get("text", "")).strip()

        if not (version or date or maintenance_person or text):
            continue

        revision_text = f"""版本：{version}
日期：{date}
維修人員：{maintenance_person}
內容：{text}""".strip()

        # 通常 revision record 很短，但仍保留 chunk 功能
        split_chunks = split_into_chunks(
            revision_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for ch_idx, ch in enumerate(split_chunks, start=1):
            row_metadata = {
                **metadata,
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
                "revision_index": idx,
                "chunk_index": ch_idx,
            }

            # 不把整包 revision_records 重複塞進每個 chunk，避免太肥
            row_metadata.pop("revision_records", None)

            chunks_out.append({
                "id": cid,
                "page": page,
                "metadata": row_metadata,
                "text": ch,

                # 兼容舊 embedding 程式用的欄位
                "source_pdf": source_pdf,
                "section": section,
                "major_title": major_title,
                "minor_title": minor_title,
                "content_type": "revision_record",
                "table_columns": ["version", "date", "maintenance_person", "text"],
                "version": version,
                "date": date,
                "maintenance_person": maintenance_person,
                "source": source_pdf or item.get("source", "未知來源"),
                "error_code": item.get("error_code", ""),
                "original_id": item.get("id", ""),
            })
            cid += 1

    return chunks_out, cid


def build_corpus(items: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    corpus = []
    cid = 0

    print(f"目前使用 chunk_size={chunk_size}, overlap={overlap}")

    for item in items:
        if not isinstance(item, dict):
            item = {
                "id": cid,
                "page": cid + 1,
                "metadata": {},
                "text": str(item)
            }

        metadata = item.get("metadata", {}) or {}

        # 只要有 revision_records 就拆，不管 content_type 有沒有寫
        revision_records = metadata.get("revision_records", []) or []

        if revision_records:
            rev_chunks, cid = build_revision_chunks(
                item=item,
                cid_start=cid,
                chunk_size=chunk_size,
                overlap=overlap
            )
            corpus.extend(rev_chunks)
            continue

        text = get_text_from_item(item)

        if not text:
            continue

        page = metadata.get("page", item.get("page", ""))
        source_pdf = metadata.get("source_pdf", item.get("source_pdf", item.get("source", "")))
        section = metadata.get("section", item.get("section", ""))
        major_title = metadata.get("major_title", item.get("major_title", ""))
        minor_title = metadata.get("minor_title", item.get("minor_title", ""))
        content_type = metadata.get("content_type", item.get("content_type", "text"))
        table_columns = metadata.get("table_columns", item.get("table_columns", []))

        chunks = split_into_chunks(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, ch in enumerate(chunks, start=1):
            row_metadata = {
                **metadata,
                "page": page,
                "source_pdf": source_pdf,
                "section": section,
                "major_title": major_title,
                "minor_title": minor_title,
                "content_type": content_type,
                "table_columns": table_columns,
                "chunk_index": chunk_index,
            }

            corpus.append({
                "id": cid,
                "page": page,
                "metadata": row_metadata,
                "text": ch,

                # 兼容舊 embedding 程式 / Qdrant payload
                "source_pdf": source_pdf,
                "section": section,
                "major_title": major_title,
                "minor_title": minor_title,
                "content_type": content_type,
                "table_columns": table_columns,
                "version": metadata.get("version", item.get("version", "")),
                "date": metadata.get("date", item.get("date", "")),
                "maintenance_person": metadata.get("maintenance_person", item.get("maintenance_person", "")),
                "source": source_pdf or item.get("source", "未知來源"),
                "error_code": item.get("error_code", ""),
                "original_id": item.get("id", ""),
            })
            cid += 1

    return corpus


def save_json(data: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已輸出：{output_path}")


def main():
    ap = argparse.ArgumentParser(description="保留 metadata 的 JSON chunk 程式")
    ap.add_argument("--input", required=True, help="輸入 JSON 檔案路徑")
    ap.add_argument("--output", default="chunked_output.json", help="輸出 JSON 檔案路徑")
    ap.add_argument("--encoding", default="", help="指定編碼，例如 utf-8 / cp950 / big5")
    ap.add_argument("--chunk-size", type=int, default=500, help="每個 chunk 最大字數")
    ap.add_argument("--overlap", type=int, default=50, help="chunk 重疊字數")
    args = ap.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(f"找不到檔案：{input_path}")

    print("=== 開始資料處理 ===")
    data = read_json_anyencoding(str(input_path), preferred_encoding=(args.encoding or None))
    items = normalize_input_items(data)

    print(f"讀到原始筆數：{len(items)}")

    corpus = build_corpus(
        items,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    print(f"切 chunk 後總筆數：{len(corpus)}")

    save_json(corpus, args.output)
    print("=== 完成 ===")


if __name__ == "__main__":
    main()
