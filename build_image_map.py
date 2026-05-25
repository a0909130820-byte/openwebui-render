import os
import json
import re

IMAGE_DIR = "static/images"
OUTPUT = "image_map.json"

image_map = {}

for file in os.listdir(IMAGE_DIR):
    match = re.search(r"_p(\d+)", file)

    if not match:
        continue

    page = match.group(1)

    if page not in image_map:
        image_map[page] = []

    image_map[page].append(file)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(image_map, f, ensure_ascii=False, indent=2)

print("完成：image_map.json")
import os
import json
import re

IMAGE_DIR = "static/images"
OUTPUT = "image_map.json"

image_map = {}

for file in os.listdir(IMAGE_DIR):
    match = re.search(r"_p(\d+)", file)

    if not match:
        continue

    page = match.group(1)

    if page not in image_map:
        image_map[page] = []

    image_map[page].append(file)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(image_map, f, ensure_ascii=False, indent=2)

print("完成：image_map.json")