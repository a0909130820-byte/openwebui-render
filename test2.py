# preprocess_json.py
# 只保留資料處理：JSON 讀取 → 清洗 → chunk 切分 → 輸出 chunked JSON

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict


TEXT_COL_CANDIDATES = ["text", "content", "page_content", "chunk"]


def read_json_items(json_path: str, preferred_encoding: str | None = None) -> List[Dict]:
    """支援多種編碼與多種 JSON 結構，回傳 [{'page': n, 'text': '...'}]"""

    def _read_text_anyencoding(path: str, preferred: str | None = None) -> tuple[str, str]:
        tries = [e for e in [preferred, "utf-8", "utf-8-sig", "cp950", "big5", "latin-1"] if e]
        with open(path, "rb") as f:
            raw = f.read()

        for enc in tries:
            try:
                return raw.decode(enc), enc
            except Exception:
                continue

        raise UnicodeDecodeError("multi-encoding", b"", 0, 1, "所有常見編碼都無法解讀")

    text, used_enc = _read_text_anyencoding(json_path, preferred_encoding)
    print(f"讀取成功，使用編碼：{used_enc}")

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"JSON 解析失敗：{e}")

    def to_page_dict(idx: int, txt: str) -> Dict:
        return {"page": idx, "text": (txt or "").strip()}

    pages: List[Dict] = []

    # 情況 1：整個 JSON 是一段文字
    if isinstance(data, str):
        pages.append(to_page_dict(1, data))
        return pages

    # 情況 2：最外層是 dict，裡面包 list
    if isinstance(data, dict):
        for key in ("items", "data", "records", "list"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    # 情況 3：list，每個 item 是 dict 或 str
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                # 先抓常見文字欄位
                found = False
                for k in TEXT_COL_CANDIDATES:
                    if k in item and isinstance(item[k], str) and item[k].strip():
                        pages.append({
                            "page": item.get("page", i),
                            "text": item[k].strip(),
                            "source": item.get("source", "未知來源"),
                            "error_code": item.get("error_code", "")
                        })
                        found = True
                        break

                # 如果沒有標準欄位，嘗試組合錯誤清單欄位
                if not found:
                    parts = []
                    for k in ["Error message", "Cause of error", "Error correction"]:
                        if k in item and isinstance(item[k], str) and item[k].strip():
                            parts.append(f"{k}: {item[k].strip()}")

                    if parts:
                        pages.append({
                            "page": item.get("page", i),
                            "text": "\n".join(parts),
                            "source": item.get("source", "錯誤清單"),
                            "error_code": item.get("error_code", "")
                        })

            elif isinstance(item, str) and item.strip():
                pages.append({
                    "page": i,
                    "text": item.strip(),
                    "source": "未知來源",
                    "error_code": ""
                })

        return pages

    # fallback
    pages.append(to_page_dict(1, str(data)))
    return pages


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """把長文字切成多個 chunks"""
    sents = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text)
    sents = [s.strip() for s in sents if s.strip()]

    chunks = []
    buf = ""

    for s in sents:
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


def build_corpus(pages: List[Dict], chunk_size: int = 800, overlap: int = 250) -> List[Dict]:
    """把 pages 轉成 chunk 後的 corpus"""
    corpus = []
    cid = 0

    for p in pages:
        if not p.get("text", "").strip():
            continue

        chunks = split_into_chunks(p["text"], chunk_size=chunk_size, overlap=overlap)

        for ch in chunks:
            corpus.append({
                "id": cid,
                "page": p.get("page", ""),
                "source": p.get("source", "未知來源"),
                "error_code": p.get("error_code", ""),
                "text": ch
            })
            cid += 1

    return corpus


def save_json(data: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已輸出：{output_path}")


def main():
    ap = argparse.ArgumentParser(description="JSON 資料處理：讀取 → chunk → 輸出")
    ap.add_argument("--input", required=True, help="輸入 JSON 檔案路徑")
    ap.add_argument("--output", default="chunked_output.json", help="輸出 JSON 檔案路徑")
    ap.add_argument("--encoding", default="", help="指定編碼，例如 utf-8 / cp950 / big5")
    ap.add_argument("--chunk-size", type=int, default=1000, help="每個 chunk 最大字數")
    ap.add_argument("--overlap", type=int, default=150, help="chunk 重疊字數")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"找不到檔案：{input_path}")

    print("=== 開始資料處理 ===")
    pages = read_json_items(str(input_path), preferred_encoding=(args.encoding or None))
    print(f"讀到頁數 / 筆數：{len(pages)}")

    corpus = build_corpus(
        pages,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"切 chunk 後總筆數：{len(corpus)}")

    save_json(corpus, args.output)
    print("=== 完成 ===")


if __name__ == "__main__":
    main()