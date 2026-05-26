import fitz
import os
import io
import json
from PIL import Image

PDFS = [
    {
        "path": "L2100 車床程式說明手冊.pdf",
        "name": "L2100_programming"
    },
    {
        "path": "L2100車床中文維護手冊(全).pdf",
        "name": "L2100_maintenance"
    },
    {
        "path": "L2100車床參數警報手冊.pdf",
        "name": "L2100_parameter_alarm"
    }
]

OUTPUT_DIR = "static/images"
IMAGE_MAP_PATH = "image_map.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_map = {}

for pdf in PDFS:
    pdf_path = pdf["path"]
    safe_name = pdf["name"]

    if not os.path.exists(pdf_path):
        print("找不到 PDF：", pdf_path)
        continue

    doc = fitz.open(pdf_path)

    print(f"處理：{pdf_path}")

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"].lower()

                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size

                if width < 120 or height < 120:
                    continue

                ratio = width / height
                if ratio < 0.3 or ratio > 5:
                    continue

                if len(image_bytes) < 5000:
                    continue

                gray = pil_img.convert("L")
                pixels = list(gray.getdata())
                avg = sum(pixels) / len(pixels)

                if avg < 15:
                    continue

                white_pixels = sum(1 for p in pixels if p > 240)
                white_ratio = white_pixels / len(pixels)

                if white_ratio < 0.35:
                    continue

                rgb = pil_img.convert("RGB")
                rgb_pixels = list(rgb.getdata())

                colorful = 0
                for r, g, b in rgb_pixels:
                    if abs(r - g) > 20 or abs(r - b) > 20 or abs(g - b) > 20:
                        colorful += 1

                color_ratio = colorful / len(rgb_pixels)
                if color_ratio > 0.25:
                    continue

                image_name = f"{safe_name}_p{page_no}_{img_index}.{image_ext}"
                image_path = os.path.join(OUTPUT_DIR, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                page_key = str(page_no)
                image_map.setdefault(page_key, [])
                image_map[page_key].append(image_name)

                print("保留圖片：", image_name)

            except Exception as e:
                print("圖片處理失敗：", e)

    doc.close()

with open(IMAGE_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(image_map, f, ensure_ascii=False, indent=2)

print("圖片抽取完成")
print("image_map.json 已更新")