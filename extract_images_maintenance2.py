import fitz
import os
import io
import json
from PIL import Image

PDF_PATH = "L2100車床中文維護手冊(全).pdf"
SAFE_NAME = "L2100_maintenance"
OUTPUT_DIR = "static/images"
IMAGE_MAP_PATH = "image_map_maintenance.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_map = {}

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"找不到 PDF：{PDF_PATH}")

doc = fitz.open(PDF_PATH)

print(f"=== 處理：{PDF_PATH} ===")

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

            # 太小圖片：通常是 icon / 小符號
            if width < 120 or height < 120:
                continue

            ratio = width / height

            # 太細長：通常不是有效手冊圖
            if ratio < 0.3 or ratio > 5:
                continue

            # 檔案太小：通常是雜圖
            if len(image_bytes) < 5000:
                continue

            gray = pil_img.convert("L")
            pixels = list(gray.getdata())
            avg = sum(pixels) / len(pixels)

            # 幾乎全黑
            if avg < 15:
                continue

            white_pixels = sum(1 for p in pixels if p > 240)
            white_ratio = white_pixels / len(pixels)

            # 白底比例太低，通常不是說明圖
            if white_ratio < 0.35:
                continue

            dark_pixels = sum(1 for p in pixels if p < 20)
            dark_ratio = dark_pixels / len(pixels)

            # 過濾黑圖
            if dark_ratio > 0.7:
                continue

            rgb = pil_img.convert("RGB")
            rgb_pixels = list(rgb.getdata())

            colorful = 0
            blue_pixels = 0

            for r, g, b in rgb_pixels:

                # 彩色比例
                if (
                    abs(r - g) > 20
                    or abs(r - b) > 20
                    or abs(g - b) > 20
                ):
                    colorful += 1

                # 藍色 UI / 裝飾圖
                if (
                    b > 150
                    and b > r + 40
                    and b > g + 40
                ):
                    blue_pixels += 1

            color_ratio = colorful / len(rgb_pixels)
            blue_ratio = blue_pixels / len(rgb_pixels)

            # 過濾大量藍色
            if blue_ratio > 0.25:
                continue

            # 過濾太彩色圖片
            if color_ratio > 0.25:
                continue

            image_name = f"{SAFE_NAME}_p{page_no}_{img_index}.{image_ext}"
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

print("\\n=== 完成 ===")
print("圖片抽取完成")
print(f"image_map 已更新：{IMAGE_MAP_PATH}")
