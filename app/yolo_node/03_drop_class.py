from pathlib import Path
import shutil
import os
from tqdm import tqdm

"""
Удалить аннотации - создает дисбаланс, неизвестный класс - сборная мусорка, трубы, аннотации
"""
REMOVE_CLASSES = [34, 36, 37]

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR.parent / '../data' /'pages'/ 'processed'
INPUT_LABELS = BASE_DIR.parent / '../data' / 'labels'
OUTPUT_DIR = BASE_DIR / 'raw_data_final'

OUTPUT_IMAGES = OUTPUT_DIR / 'images'
OUTPUT_LABELS = OUTPUT_DIR / 'labels'
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

stats = {
    'total': 0,
    'removed': 0,
    'kept': 0
}
removed_by_class = {c: 0 for c in REMOVE_CLASSES}
label_files = list(INPUT_LABELS.glob('*.txt'))

for label_file in tqdm(label_files, desc='Processing_labels', unit='file'):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    stats['total'] += len(lines)

    filtered_lines = []
    for line in lines:
        class_id = int(line.split()[0])

        if class_id in REMOVE_CLASSES:
            stats['removed'] += 1
            removed_by_class[class_id] += 1
        else:
            filtered_lines.append(line)
            stats['kept'] += 1

    if not filtered_lines:
        continue

    with open(OUTPUT_LABELS / label_file.name, 'w') as f:
        f.writelines(filtered_lines)

        img_name = label_file.stem + '.png'
        img_path = INPUT_PATH / img_name

        if img_path.exists():
            shutil.copy(img_path, OUTPUT_IMAGES / img_name)
        else:
            img_path = INPUT_PATH / (label_file.stem + ".jpg")
            if img_path.exists():
                shutil.copy(img_path, OUTPUT_IMAGES / img_name)
            else:
                print(f"Изображение не найдено: {img_name}")