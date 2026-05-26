import fitz
import os
import json
import re

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None


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


def looks_broken(text: str) -> bool:
    if not text:
        return True

    bad_markers = ["æ", "è", "ä", "å", "ç", "Ã", "Â", "ï¼", "ï"]
    bad_count = sum(text.count(x) for x in bad_markers)

    chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")

    return bad_count > 10 and chinese_count < 20


def fix_mojibake(text: str) -> str:
    if not text:
        return text

    try:
        fixed = text.encode("latin1").decode("utf-8")
        if not looks_broken(fixed):
            return fixed
    except Exception:
        pass

    try:
        fixed = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        if not looks_broken(fixed):
            return fixed
    except Exception:
        pass

    return text


def extract_with_pymupdf(pdf_path, page_index):
    doc = fitz.open(pdf_path)
    text = doc[page_index].get_text("text")
    doc.close()
    return text or ""


def extract_with_pypdf(pdf_path, page_index):
    if PdfReader is None:
        return ""

    reader = PdfReader(pdf_path)
    page = reader.pages[page_index]
    return page.extract_text() or ""


def extract_with_pdfplumber(pdf_path, page_index):
    if pdfplumber is None:
        return ""

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        return page.extract_text() or ""


def extract_page_text(pdf_path, page_index):
    candidates = []

    for extractor in [
        extract_with_pymupdf,
        extract_with_pypdf,
        extract_with_pdfplumber
    ]:
        try:
            text = extractor(pdf_path, page_index)
            text = fix_mojibake(text)
            candidates.append(text)
        except Exception:
            pass

    candidates = [c for c in candidates if c and c.strip()]

    if not candidates:
        return ""

    candidates.sort(
        key=lambda t: (
            looks_broken(t),
            -sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff"),
            -len(t)
        )
    )

    return candidates[0]


def detect_codes(text):
    patterns = [
        r"\bG\d{1,3}(?:\.\d)?\b",
        r"\bM\d{1,3}\b",
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


def clean_text(text):
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


records = []
record_id = 1

for pdf in PDFS:
    pdf_path = pdf["path"]
    manual_type = pdf["manual_type"]

    if not os.path.exists(pdf_path):
        print("找不到：", pdf_path)
        continue

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    source_file = os.path.basename(pdf_path)

    for page_index in range(page_count):
        page_no = page_index + 1

        text = extract_page_text(pdf_path, page_index)
        text = clean_text(text)

        if not text:
            continue

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

        print(f"完成 {source_file} 第 {page_no} 頁")

        record_id += 1


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"完成：{OUTPUT_JSON}")
print(f"共 {len(records)} 筆")