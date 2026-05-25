import fitz
import os
import io
from PIL import Image

PDFS = [
    "L2100 車床程式說明手冊.pdf",
    "L2100車床中文維護手冊(全).pdf",
    "L2100車床參數警報手冊.pdf"
]

OUTPUT_DIR = "static/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for pdf_path in PDFS:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    print(f"處理：{pdf_name}")

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

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

                image_name = f"{pdf_name}_p{page_index+1}_{img_index+1}.{image_ext}"
                image_path = os.path.join(OUTPUT_DIR, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                print("保留圖片：", image_name)

            except Exception as e:
                print("圖片處理失敗：", e)

print("圖片抽取完成")