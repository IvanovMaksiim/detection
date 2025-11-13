from pathlib import Path
import random
import shutil
"""
Stratificated split. 
target is one exp of class in train and one exp of class in val
"""
random.seed(42)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / 'dataset_two_stage' / 'trainval_tiles'
OUTPUT_DIR = BASE_DIR / 'dataset_two_stage'

VAL_RATIO = 0.3

labels_dir = INPUT_DIR / 'labels'
images_dir = INPUT_DIR / 'images'
masks_dir = INPUT_DIR / 'forbidden_masks'


all_tiles = [f.stem for f in labels_dir.glob('*.txt')]
print(f"Всего тайлов: {len(all_tiles)}")

tile_to_classes = {}
for tile in all_tiles:
    label_path = labels_dir / f'{tile}.txt'
    classes = set()
    if label_path.exists():
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    tile_to_classes[tile] = classes

all_classes = set()
for cls_cls in tile_to_classes.values():
    all_classes.update(cls_cls)

# Stratified split
train_tiles = set()
val_tiles = set()
remaining_tiles = set(all_tiles)

for cls_id in all_classes:
    cls_tiles = [t for t in all_tiles if cls_id in tile_to_classes[t]]

    train_chosen = None
    for t in cls_tiles:
        if t not in train_tiles and t not in val_tiles:
            train_chosen = t
            break
    if train_chosen:
        train_tiles.add(train_chosen)
        remaining_tiles.discard(train_chosen)

    val_chosen = None
    for t in cls_tiles:
        if t not in train_tiles and t not in val_tiles:
            val_chosen = t
            break
    if val_chosen:
        val_tiles.add(val_chosen)
        remaining_tiles.discard(val_chosen)

target_val_count = int(len(all_tiles)*VAL_RATIO)
remaining_list = list(remaining_tiles)
random.shuffle(remaining_list)

for t in remaining_list:
    if len(val_tiles) >= target_val_count:
        break
    val_tiles.add(t)
    remaining_tiles.discard(t)

train_tiles.update(remaining_tiles)

print(f"\nTrain: {len(train_tiles)} tiles ({len(train_tiles) / len(all_tiles) * 100:.1f}%)")
print(f"Val:   {len(val_tiles)} tiles ({len(val_tiles) / len(all_tiles) * 100:.1f}%)")

# save imgs + labels in right place
for split_name, tiles in [('train', train_tiles), ('val', val_tiles)]:
    images_out = OUTPUT_DIR / split_name / 'images'
    labels_out = OUTPUT_DIR / split_name / 'labels'
    mask_out = OUTPUT_DIR / split_name / 'forbidden_mask'

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    copied_imgs = 0
    copied_labels = 0
    copied_masks = 0

    for tile_name in tiles:
        src_img = images_dir / f'{tile_name}.png'
        if src_img.exists():
            shutil.copy(src_img, images_out / f'{tile_name}.png')
            copied_imgs += 1

        src_label = labels_dir / f'{tile_name}.txt'
        if src_label.exists():
            shutil.copy(src_label, labels_out / f'{tile_name}.txt')
            copied_labels += 1

        src_mask = masks_dir / f'{tile_name}.png'
        if src_mask.exists():
            shutil.copy(src_mask, mask_out / f'{tile_name}.png')
            copied_masks += 1

    print(f"\n{split_name.upper()}:")
    print(f"   Изображений: {copied_imgs}")
    print(f"   Labels: {copied_labels}")
    print(f"   Масок: {copied_masks}")

# Statistics
train_classes = set()
val_classes = set()

for tile in train_tiles:
    train_classes.update(tile_to_classes[tile])
for tile in val_tiles:
    val_classes.update(tile_to_classes[tile])

print(f"Классов в Train: {len(train_classes)}")
print(f"Классов в Val:   {len(val_classes)}")

if len(train_classes) == len(all_classes):
    print(f"Все {len(all_classes)} классов в Train!")
else:
    missing = all_classes - train_classes
    print(f"Отсутствуют в Train: {sorted(missing)}")

if len(val_classes) == len(all_classes):
    print(f"Все {len(all_classes)} классов в Val!")
else:
    missing = all_classes - val_classes
    print(f"Отсутствуют в Val: {sorted(missing)}")