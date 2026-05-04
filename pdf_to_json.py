import fitz
import json
import re
import argparse
from pathlib import Path


ERROR_CODE_PATTERN = r"[A-Z0-9]{3}-[A-Z0-9]{4}"
ERROR_CODE_RE = re.compile(ERROR_CODE_PATTERN)


def normalize_error_codes(text):
    text = text.upper()

    # 修正常見不同破折號
    text = text.replace("‐", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # 修 231 - A043 / 231-\nA043 / 231\n-\nA043
    text = re.sub(
        r"([A-Z0-9]{3})\s*-\s*([A-Z0-9]{4})",
        r"\1-\2",
        text
    )

    return text


def clean_text(text):
    text = text.replace("\r", "\n")
    text = normalize_error_codes(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_error_code(text):
    match = ERROR_CODE_RE.search(text)
    return match.group(0) if match else ""


def get_page_text(page):
    # 用 blocks 比單純 text 更適合表格 PDF
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    texts = []

    for b in blocks:
        if len(b) >= 5:
            texts.append(b[4])

    return "\n".join(texts)


def pdf_to_json_structured(pdf_path, output_path):
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        print(f"找不到 PDF：{pdf_path}")
        return

    doc = fitz.open(pdf_path)

    data = []
    missing_pages = []
    idx = 1

    for page_num, page in enumerate(doc, start=1):
        text = get_page_text(page)
        text = clean_text(text)

        codes = ERROR_CODE_RE.findall(text)

        if not codes:
            missing_pages.append(page_num)
            continue

        # 在每個錯誤碼前切段
        blocks = re.split(rf"\n(?={ERROR_CODE_PATTERN})", text)

        for block in blocks:
            block = clean_text(block)
            error_code = extract_error_code(block)

            if error_code:
                data.append({
                    "id": idx,
                    "page": page_num,
                    "source": "未知來源",
                    "error_code": error_code,
                    "text": block
                })
                idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open("missing_pages.txt", "w", encoding="utf-8") as f:
        for p in missing_pages:
            f.write(str(p) + "\n")

    print("完成")
    print(f"共輸出 {len(data)} 筆")
    print(f"輸出檔案：{output_path}")
    print("未抓到錯誤碼的頁數已存到 missing_pages.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="error_codes.json")

    args = parser.parse_args()

    pdf_to_json_structured(args.input, args.output)