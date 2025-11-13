import os
import numpy as np
from PIL import Image
from collections import defaultdict
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent

# path for 01
# IMAGES_DIR = BASE_DIR.parent / "../data" / "pages"
# LABELS_DIR = BASE_DIR.parent / "../data" / "labels"
#
# CSV_PATH = BASE_DIR / "./statistic" / "class_summary_raw.csv"

# путь для 04
IMAGES_DIR = BASE_DIR / "./raw_data_final" / "images"
LABELS_DIR = BASE_DIR / "./raw_data_final" / "labels"

CSV_PATH = BASE_DIR / "./statistic" / "class_summary_clean.csv"

NAMES_PATH = BASE_DIR.parent / "../data" / "odj.names"

RARE_THRESHOLD = 50
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

if os.path.exists(NAMES_PATH):
    with open(NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]
else:
    class_names = []



def get_image_size(img_name):
    """Расчитать размер изображения"""

    for ext in ('.png', '.jpg', 'jpeg'):
        path = os.path.join(IMAGES_DIR, os.path.splitext(img_name)[0] + ext)
        if os.path.exists(path):
            with Image.open(path) as img:
                return img.size
    return None


def get_class_name(cls_id):
    """ Возвращает имя класса или его номер, если имени нет"""
    if 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return f"id_{cls_id}"


# == Анализ ==
class_stats = defaultdict(list)

for file in os.listdir(LABELS_DIR):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(LABELS_DIR, file)
    image_size = get_image_size(file)
    if not image_size:
        print(f'Нет изображения для {file}')

    img_w, img_h = image_size

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, _, _, w, h = map(float, parts)
            cls_id = int(cls_id)
            w_px = w * img_w
            h_px = h * img_h
            avg_size = np.mean([w_px, h_px])
            class_stats[cls_id].append(avg_size)

# === Подсчёт ===
summary = []
for cls_id, sizes in sorted(class_stats.items()):
    count = len(sizes)
    size_min, size_max = np.min(sizes), np.max(sizes)
    summary.append((cls_id, count, round(size_min, 1), round(size_max, 1)))

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["class_id", "class_name", "count", "size_min_px", "size_max_px", "is_rare"])

    for cls_id, count, size_min, size_max in summary:
        cname = get_class_name(cls_id)
        rare_flag = "YES" if count < RARE_THRESHOLD else "NO"
        writer.writerow([cls_id, cname, count, size_min, size_max, rare_flag])

print(f'\n CSV save: {CSV_PATH}')
