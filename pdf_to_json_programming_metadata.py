import fitz
import os
import json
import re
from pathlib import Path

# ===============================
# 設定
# ===============================
PDF_PATH = "L2100 車床程式說明手冊.pdf"
OUTPUT_JSON = "data/l2100_programming_metadata.json"
IMAGE_DIR = "static/images"

os.makedirs("data", exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ===============================
# 基本清理
# ===============================
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
        r"\bT\d{2,4}\b",
        r"\bS\d+\b",
        r"\bF\d+(?:\.\d+)?\b",
        r"\bX-?\d+(?:\.\d+)?\b",
        r"\bY-?\d+(?:\.\d+)?\b",
        r"\bZ-?\d+(?:\.\d+)?\b",
    ]

    codes = []
    for p in patterns:
        codes.extend(re.findall(p, text, flags=re.IGNORECASE))

    # 保留順序並去重
    return list(dict.fromkeys(codes))


# ===============================
# 從目錄頁抓小標題
# ===============================
def parse_toc(doc):
    toc_text = ""

    # 程式說明手冊目錄在前兩頁
    for i in range(min(2, len(doc))):
        toc_text += "\n" + doc[i].get_text("text")

    normalized = re.sub(r"\s+", " ", toc_text)

    items = []

    # 例如：1.3 圓弧插值(G02/G03) .............. 7
    pattern = r"(\d+\.\d+)\s+(.+?)\s+\.{3,}\s+(\d+)"

    for m in re.finditer(pattern, normalized):
        number = m.group(1).strip()
        title = m.group(2).strip()
        page = int(m.group(3))

        items.append({
            "number": number,
            "title": title,
            "start_page": page,
            "minor_title": f"{number} {title}",
        })

    # 補 end_page
    for i, item in enumerate(items):
        if i + 1 < len(items):
            item["end_page"] = items[i + 1]["start_page"] - 1
        else:
            item["end_page"] = 9999

    return items


def get_major_title(page_no: int, text: str) -> str:
    # 依照手冊結構粗分大章節
    if page_no <= 111:
        return "一、G 碼指令"
    elif page_no <= 118:
        return "二、M 輔助機能"
    else:
        return "三、巨集程式"


def find_minor_title(page_no: int, toc_items):
    for item in toc_items:
        if item["start_page"] <= page_no <= item["end_page"]:
            return item["minor_title"]
    return ""


def detect_content_type(text: str, minor_title: str) -> str:
    t = text or ""

    if "TYPE A" in t and "TYPE B" in t and "TYPE C" in t:
        return "table"

    if "一覽表" in t or "機能表" in t:
        return "table"

    if "指令格式" in t or "範例" in t or "注意事項" in t:
        return "instruction"

    return "text"


# ===============================
# 抽圖片
# 每一頁圖片都存到 static/images
# JSON 裡存 /static/images/xxx.png
# ===============================
def extract_images_for_page(doc, page_index: int, source_stem: str):
    page = doc[page_index]
    page_no = page_index + 1
    image_paths = []

    images = page.get_images(full=True)

    for img_index, img in enumerate(images, start=1):
        xref = img[0]

        try:
            pix = fitz.Pixmap(doc, xref)

            # CMYK 或特殊色彩空間轉 RGB
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            filename = f"{source_stem}_p{page_no}_{img_index}.png"
            save_path = os.path.join(IMAGE_DIR, filename)

            pix.save(save_path)
            pix = None

            image_paths.append(f"/static/images/{filename}")

        except Exception as e:
            print(f"圖片抽取失敗 page={page_no}, image={img_index}: {e}")

    return image_paths


# ===============================
# 主流程
# ===============================
def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"找不到 PDF：{PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    source_file = os.path.basename(PDF_PATH)
    source_stem = Path(PDF_PATH).stem.replace(" ", "_")

    toc_items = parse_toc(doc)

    records = []
    record_id = 1

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]

        text = clean_text(page.get_text("text"))

        if not text:
            continue

        major_title = get_major_title(page_no, text)
        minor_title = find_minor_title(page_no, toc_items)

        content_type = detect_content_type(text, minor_title)
        codes = detect_codes(text)

        images = extract_images_for_page(
            doc=doc,
            page_index=page_index,
            source_stem=source_stem
        )

        metadata = {
            "page": page_no,
            "source_pdf": source_file,
            "manual_type": "programming",
            "section": minor_title or major_title,
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

        print(f"完成 page={page_no}, minor_title={minor_title}, images={len(images)}")
        record_id += 1

    doc.close()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    print(f"輸出 JSON：{OUTPUT_JSON}")
    print(f"總筆數：{len(records)}")


if __name__ == "__main__":
    main()
