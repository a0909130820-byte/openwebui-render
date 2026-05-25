import fitz
import os
import json
import re

PDFS = [
    {
        "path": "L2100 車床程式說明手冊.pdf",
        "manual_type": "programming"
    },
    {
        "path": "L2100車床中文維護手冊(全).pdf",
        "manual_type": "maintenance"
    },
    {
        "path": "L2100車床參數警報手冊.pdf",
        "manual_type": "parameter_alarm"
    }
]

IMAGE_MAP_PATH = "image_map.json"
OUTPUT_JSON = "data/l2100_manuals.json"

os.makedirs("data", exist_ok=True)

if os.path.exists(IMAGE_MAP_PATH):
    with open(IMAGE_MAP_PATH, "r", encoding="utf-8") as f:
        IMAGE_MAP = json.load(f)
else:
    IMAGE_MAP = {}

def detect_codes(text):
    patterns = [
        r"\bG\d{2,3}(?:\.\d)?\b",
        r"\bM\d{2,3}\b",
        r"\bINT\s*\d+\b",
        r"\bMOT\s*\d+\b",
        r"\bOP\s*\d+\b",
        r"\bRTEX\s*\d+\b",
        r"\bEtherCAT\b",
        r"\b\d{4}\b"
    ]

    codes = []
    for p in patterns:
        codes.extend(re.findall(p, text, flags=re.IGNORECASE))

    return list(dict.fromkeys(codes))

def get_images_for_page(page):
    page_key = str(page)
    images = []

    if page_key in IMAGE_MAP:
        for filename in IMAGE_MAP[page_key]:
            images.append(f"/static/images/{filename}")

    return images

records = []
record_id = 1

for pdf in PDFS:
    pdf_path = pdf["path"]
    manual_type = pdf["manual_type"]

    if not os.path.exists(pdf_path):
        print("找不到：", pdf_path)
        continue

    doc = fitz.open(pdf_path)
    source_file = os.path.basename(pdf_path)

    for page_index in range(len(doc)):
        page_no = page_index + 1
        text = doc[page_index].get_text("text").strip()

        if not text:
            continue

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0][:80] if lines else ""

        records.append({
            "id": record_id,
            "source_file": source_file,
            "manual_type": manual_type,
            "page": page_no,
            "title": title,
            "codes": detect_codes(text),
            "text": text,
            "images": get_images_for_page(page_no)
        })

        record_id += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"完成：{OUTPUT_JSON}")
print(f"共 {len(records)} 筆")