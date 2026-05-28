import fitz
import os
import json
import re
from pathlib import Path

# =========================================================
# 參數警報手冊 PDF → JSON
# 功能：
# 1. 只抽文字
# 2. 不重新抽圖片
# 3. 從 static/images 裡依照檔名規則加入圖片 metadata
#
# 圖片檔名建議格式：
# static/images/L2100_parameter_alarm_p5_1.png
# static/images/L2100_parameter_alarm_p5_2.png
# =========================================================

PDF_PATH = "L2100車床參數警報手冊.pdf"
OUTPUT_JSON = "data/l2100_parameter_alarm_metadata.json"

IMAGE_DIR = "static/images"
IMAGE_PREFIX = "L2100_parameter_alarm"

os.makedirs("data", exist_ok=True)


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_codes(text: str):
    patterns = [
        r"\bG\d{1,3}(?:\.\d)?\b",
        r"\bM\d{1,3}\b",
        r"\bINT\s*\d+\b",
        r"\bMOT\s*\d+\b",
        r"\bOP\s*\d+\b",
        r"\bRTEX\s*\d+\b",
        r"\bEtherCAT\b",
        r"\b\d{3,4}[-－][A-Za-z0-9]{1,6}\b",
        r"\b\d{4}\b",
        r"\bP\d{1,5}\b",
        r"\bPRM\s*\d+\b",
        r"\bALM\s*\d+\b",
    ]

    codes = []
    for p in patterns:
        codes.extend(re.findall(p, text, flags=re.IGNORECASE))

    return list(dict.fromkeys(codes))


def get_images_for_page(page_no: int):
    """
    從 static/images 掃描該頁圖片。
    不重新抽圖片，只把圖片路徑寫到 metadata.images。
    """
    images = []
    image_dir = Path(IMAGE_DIR)

    if not image_dir.exists():
        return images

    patterns = [
        f"{IMAGE_PREFIX}_p{page_no}_*.png",
        f"{IMAGE_PREFIX}_p{page_no}_*.jpg",
        f"{IMAGE_PREFIX}_p{page_no}_*.jpeg",
    ]

    files = []

    for pat in patterns:
        files.extend(image_dir.glob(pat))

    for file in sorted(files, key=lambda x: x.name):
        images.append(f"/static/images/{file.name}")

    return images


def guess_major_title(page_no: int, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        if 2 <= len(line) <= 60:
            return line

    return "L2100 車床參數警報手冊"


def guess_minor_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:12]:
        if re.match(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+[、.])", line):
            return line[:80]

    # 優先抓警報/參數相關行
    for line in lines[:15]:
        if "警報" in line or "參數" in line or "異常" in line:
            return line[:80]

    if lines:
        return lines[0][:80]

    return ""


def detect_content_type(text: str) -> str:
    if not text:
        return "text"

    if "警報" in text or "ALARM" in text.upper() or "異常" in text:
        return "alarm"

    if "參數" in text or "PARAMETER" in text.upper():
        return "parameter"

    if "表" in text and ("項目" in text or "說明" in text or "名稱" in text):
        return "table_or_list"

    if "原因" in text or "處理" in text or "解除" in text or "對策" in text:
        return "troubleshooting"

    return "text"


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"找不到 PDF：{PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    source_file = os.path.basename(PDF_PATH)

    records = []
    record_id = 1

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]

        text = clean_text(page.get_text("text"))

        if not text:
            continue

        major_title = guess_major_title(page_no, text)
        minor_title = guess_minor_title(text)
        section = minor_title or major_title
        content_type = detect_content_type(text)
        codes = detect_codes(text)
        images = get_images_for_page(page_no)

        metadata = {
            "page": page_no,
            "source_pdf": source_file,
            "manual_type": "parameter_alarm",
            "section": section,
            "major_title": major_title,
            "minor_title": minor_title,
            "content_type": content_type,
            "codes": codes,
            "images": images,
        }

        records.append({
            "id": record_id,
            "page": page_no,
            "metadata": metadata,
            "text": text
        })

        print(f"完成 page={page_no}, images={len(images)}, title={minor_title}")

        record_id += 1

    doc.close()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    print(f"輸出 JSON：{OUTPUT_JSON}")
    print(f"總筆數：{len(records)}")
    print("注意：圖片本體沒有寫進 JSON；metadata.images 只保存圖片路徑。")


if __name__ == "__main__":
    main()
