import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from collections import defaultdict

"""
Copy-Paste augmentation for rare class

Rules paste:
- 70% - completely in background
- 20% - not all visible (80-90% object)
- 10% - strong occlusion

Augmentation
- turns: 0, 90, 180, 270
- color: slight variations in brightness
- line thickness
- adaptive binarization

Defense:
- masks_truba
- mask_annotation
"""

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).parent
INPUT_TEST = BASE_DIR / 'raw_data_split_two_stage' / 'test'
DATASET_DIR = BASE_DIR / 'dataset_two_stage'
TRAIN_IMAGES = DATASET_DIR / 'train' / 'images'
TRAIN_LABELS = DATASET_DIR / 'train' / 'labels'
FORBIDDEN_MASKS = DATASET_DIR / "train" / "forbidden_masks"
OUTPUT_DIR = DATASET_DIR / "train_augmented"
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"
OUTPUT_TEST = DATASET_DIR / 'test'
RARE_CLASSES = [
    7,  # klapan_obratn_seroprivod
    9,  # regulator_seroprivod
    10,  # armatura_membr_electro
    12,  # ventilaytor
    14,  # condensatootvod
    16,  # vodostruiniy_nasos
    18,  # zaglushka
    19,  # gidrozatvor
    23,  # separator
    24,  # kapleulov
    26,  # redukcion_ustr
    27,  # bistro_redukc_ustr
    28,  # separator_paro
    29,  # dearator
    30,  # silfonnii_kompensator
    31,  # electronagrevat
    32,  # smotrowoe_steclo
]

CRITICAL_CLASSES = [10, 12, 16, 19, 23, 26, 28, 29, 31]

TARGET_COUNTS = {}
for cls in RARE_CLASSES:
    if cls in CRITICAL_CLASSES:
        TARGET_COUNTS[cls] = 200
    else:
        TARGET_COUNTS[cls] = 150

INSERTION_RULES = {
    'full_visible': 0.70,
    'partial_80_90': 0.20,
    'occlusion': 0.10
}

MAX_ATTEMPTS = 50
FORBIDDEN_THRESHOLD = 0.10
MIN_OBJECT_SIZE = 20

SYNTHETIC_TILE_SIZE = 1280
OBJECTS_PER_SYNTHETIC_TILE = (3, 8)
MAX_SYNTHETIC_TILES = 300


# Creation of node bank
def yolo_to_bbox(yolo_coords, img_w, img_h):
    '''
    Conversion YOLO to px coordinate
    :param yolo_coords: [0,1]
    :param img_w:
    :param img_h:
    :return: pixel coordinates
    '''

    x_center, y_center, w, h = yolo_coords
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)

    clamp_x1 = max(0, min(img_w - 1, x1))
    clamp_y1 = max(0, min(img_h - 1, y1))
    clamp_x2 = max(0, min(img_w - 1, x2))
    clamp_y2 = max(0, min(img_h - 1, y2))

    return clamp_x1, clamp_y1, clamp_x2, clamp_y2


def collect_objects():
    '''
    collect everyone objects of rare classes from train
    :return: objects = {class_id = (img - after cv2,
                                    bbox - pixel coordinates,
                                    tile_name)}
    '''

    objects = {cls: [] for cls in RARE_CLASSES}

    for label_file in tqdm(list(TRAIN_LABELS.glob('*.txt')), desc='Scanning'):
        img_file = TRAIN_IMAGES / f"{label_file.stem}.png"

        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                if class_id not in RARE_CLASSES:
                    continue

                yolo_bbox = tuple(map(float, parts[1:5]))
                bbox = yolo_to_bbox(yolo_bbox, w, h)

                objects[class_id].append((img, bbox, label_file.stem))

    print(f"\n Найдено объектов:")
    for cls in sorted(RARE_CLASSES):
        count = len(objects[cls])
        print(f"  Класс {cls:2d}: {count:3d} объектов")

    return objects


# Augmentation
def extract_object_with_mask(img, bbox, padding=0.05):
    '''
    Extract object with white background. Opening and Closing processing/
    :param img:
    :param bbox:
    :param padding: indent around object (relatively)
    :return: crop: img object,
             mask: bin mask(0, 255)
    '''

    x1, y1, x2, y2 = bbox
    h, w = img.shape

    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None, None

    mask = (crop < 200).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    crop[mask == 0] = 255

    return crop, mask


def augment_rotation(crop, mask, angle):
    '''
    Rotation 0, 90, 180, 270
    :param crop:
    :param mask:
    :param angle:
    :return: rotation crop, rotation mask
    '''

    if angle == 0:
        return crop, mask

    if angle == 90:
        crop_rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        mask_rot = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        crop_rot = cv2.rotate(crop, cv2.ROTATE_180)
        mask_rot = cv2.rotate(mask, cv2.ROTATE_180)
    elif angle == 270:
        crop_rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask_rot = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return crop, mask

    return crop_rot, mask_rot


def augment_brightness(crop, delta_range):
    '''
    Change brightness
    :param crop:
    :param delta_range: limits of changes
    :return: aug_crop
    '''
    delta = random.randint(*delta_range)
    crop_aug = np.clip(crop.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    return crop_aug


def augment_thickness(crop, mask, operation, kernel_size):
    '''
    Change thickness of line. Variants are erode, dilate.
    :param crop:
    :param mask:
    :param operation: erode, dilate
    :param kernel_size:
    :return: crop_aug, mask_aug
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation == 'erode':
        mask_aug = cv2.erode(mask, kernel, iterations=1)
    elif operation == 'dilate':
        mask_aug = cv2.dilate(mask, kernel, iterations=1)
    else:
        mask_aug = mask

    crop_aug = crop.copy()
    crop_aug[mask_aug == 0] = 255

    return crop_aug, mask_aug


def augment_binarization(crop, mask):
    '''
    Adaptive binarization
    :param crop:
    :param mask:
    :return: crop_bin
    '''

    crop_bin = crop.copy()

    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    crop_bin[mask > 0] = binary[mask > 0]

    return crop_bin


def apply_augmentations(crop, mask):
    '''
    Use random augmentation to crop. Turn 100%, brightness 50%, thickness 30%, binarization 20%
    :param crop:
    :param mask:
    :return: aug_crop, aug_mask
    '''

    angle = random.choice([0, 90, 180, 270])
    crop_aug, mask_aug = augment_rotation(crop, mask, angle)

    if random.random() < 0.5:
        crop_aug = augment_brightness(crop_aug, delta_range=(-15, 15))

    if random.random() < 0.3:
        operation = random.choice(['erode', 'dilate'])
        kernel_size = random.choice([2, 3])
        crop_aug, mask_aug = augment_thickness(crop_aug, mask_aug, operation, kernel_size)

    if random.random() < 0.2:
        crop_aug = augment_binarization(crop_aug, mask_aug)

    return crop_aug, mask_aug


def load_forbidden_mask(tile_name):
    '''
    Download mask prohibited areas
    :param tile_name:
    :return: mask
    '''
    mask_path = FORBIDDEN_MASKS / f'{tile_name}_forbidden.png'

    if not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    return mask


def calculate_iou(bbox1, bbox2):
    '''
    Calculate IoU -> YOLO format
    :param bbox1:
    :param bbox2:
    :return: float IoU
    '''
    b1_x1 = bbox1[0] - bbox1[2] / 2
    b1_y1 = bbox1[1] - bbox1[3] / 2
    b1_x2 = bbox1[0] + bbox1[2] / 2
    b1_y2 = bbox1[1] + bbox1[3] / 2

    b2_x1 = bbox2[0] - bbox2[2] / 2
    b2_y1 = bbox2[1] - bbox2[3] / 2
    b2_x2 = bbox2[0] + bbox2[2] / 2
    b2_y2 = bbox2[1] + bbox2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    IoU_result = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return IoU_result


def find_position_full_visible(tile, crop, forbidden_mask, existing_bboxes):
    '''
    Find position where object full visible (70%)
    :param tile:
    :param crop:
    :param forbidden_mask:
    :param existing_bboxes:
    :return: (x, y, bbox_yolo) or None
    '''
    tile_h, tile_w = tile.shape
    crop_h, crop_w = crop.shape

    for _ in range(MAX_ATTEMPTS):
        if crop_w >= tile_w or crop_h >= tile_h:
            return None

        x = random.randint(0, tile_w - crop_w)
        y = random.randint(0, tile_h - crop_h)

        if forbidden_mask is not None:
            mask_region = forbidden_mask[y:y + crop_h, x:x + crop_w]
            forbidden_ratio = (mask_region > 0).sum() / mask_region.size

            if forbidden_ratio > FORBIDDEN_THRESHOLD:
                continue

        new_bbox = (
            (x + crop_w / 2) / tile_w,
            (y + crop_h / 2) / tile_h,
            crop_w / tile_w,
            crop_h / tile_h
        )

        overlap = any(calculate_iou(new_bbox, ex) > 0.05 for ex in existing_bboxes)
        if overlap:
            continue

        return x, y, new_bbox

    return None


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    '''
    Calculate bbox to YOLO format
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param img_w:
    :param img_h:
    :return: x_center, y_center, w, h
    '''
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h

    return x_center, y_center, w, h


def find_position_partial(tile, crop, forbidden_mask, existing_bboxes, visibility_range=(0.8, 0.9)):
    '''
    find position where object partial (20%)
    :param tile:
    :param crop:
    :param forbidden_mask:
    :param existing_bboxes:
    :param visibility_range:
    :return: (x, y, visible_bbox_yolo) or None
    '''
    tile_h, tile_w = tile.shape
    crop_h, crop_w = crop.shape

    for _ in range(MAX_ATTEMPTS):

        target_visibility = random.uniform(*visibility_range)

        side = random.choice(['top', 'bottom', 'left', 'right'])

        if side == 'left':
            cutoff = int(crop_w * (1 - target_visibility))
            x = random.randint(-cutoff, 0)
            y = random.randint(0, max(0, tile_h - crop_h))
        elif side == 'right':
            cutoff = int(crop_w * (1 - target_visibility))
            x = random.randint(tile_w - crop_w, tile_w - crop_w + cutoff)
            y = random.randint(0, max(0, tile_h - crop_h))
        elif side == 'top':
            cutoff = int(crop_h * (1 - target_visibility))
            x = random.randint(0, max(0, tile_w - crop_w))
            y = random.randint(-cutoff, 0)
        else:
            cutoff = int(crop_h * (1 - target_visibility))
            x = random.randint(0, max(0, tile_w - crop_w))
            y = random.randint(tile_h - crop_h, tile_h - crop_h + cutoff)

        vis_x1 = max(0, x)
        vis_y1 = max(0, y)
        vis_x2 = min(tile_w, x + crop_w)
        vis_y2 = min(tile_h, y + crop_h)

        vis_w = vis_x2 - vis_x1
        vis_h = vis_y2 - vis_y1

        if vis_w < MIN_OBJECT_SIZE or vis_h < MIN_OBJECT_SIZE:
            continue

        if forbidden_mask is not None:
            mask_region = forbidden_mask[vis_y1:vis_y2, vis_x1:vis_x2]
            if mask_region.size > 0:
                forbidden_ratio = (mask_region > 0).sum() / mask_region.size
                if forbidden_ratio > FORBIDDEN_THRESHOLD:
                    continue

        visible_bbox = bbox_to_yolo(vis_x1, vis_y1, vis_x2, vis_y2, tile_w, tile_h)

        overlap = any(calculate_iou(visible_bbox, ex) > 0.15 for ex in existing_bboxes)
        if overlap:
            continue

        return x, y, visible_bbox

    return None


def find_position_occlusion(tile, crop, forbidden_mask, existing_bboxes):
    '''
    Find position with occlusion
    :param tile:
    :param crop:
    :param forbidden_mask:
    :param existing_bboxes:
    :return: (x, y, visible_bbox_yolo) or None
    '''
    tile_h, tile_w = tile.shape
    crop_h, crop_w = crop.shape

    for _ in range(MAX_ATTEMPTS):

        x = random.randint(-crop_w // 2, tile_w - crop_w // 2)
        y = random.randint(-crop_h // 2, tile_h - crop_h // 2)

        vis_x1 = max(0, x)
        vis_y1 = max(0, y)
        vis_x2 = min(tile_w, x + crop_w)
        vis_y2 = min(tile_h, y + crop_h)

        vis_w = vis_x2 - vis_x1
        vis_h = vis_y2 - vis_y1

        if vis_w < MIN_OBJECT_SIZE or vis_h < MIN_OBJECT_SIZE:
            continue

        if forbidden_mask is not None:
            mask_region = forbidden_mask[vis_y1:vis_y2, vis_x1:vis_x2]
            if mask_region.size > 0:
                forbidden_ratio = (mask_region > 0).sum() / mask_region.size
                if forbidden_ratio > FORBIDDEN_THRESHOLD * 2:  # более мягкий порог
                    continue

        visible_bbox = bbox_to_yolo(vis_x1, vis_y1, vis_x2, vis_y2, tile_w, tile_h)

        iou_values = [calculate_iou(visible_bbox, ex) for ex in existing_bboxes]
        max_iou = max(iou_values) if iou_values else 0

        if random.random() < 0.5:
            if max_iou < 0.10 or max_iou > 0.24:
                continue
        else:
            if max_iou < 0.25 or max_iou > 0.40:
                continue

        if max_iou > 0.50:
            continue

        return x, y, visible_bbox

    return None


def insert_object(tile, crop, mask, x, y):
    """
    insert object to tiles

    Args:
        tile:
        crop:
        mask:
        x, y: position insert (may be negative if not 100% visible)
    return: tile
    """
    tile_h, tile_w = tile.shape
    crop_h, crop_w = crop.shape

    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(crop_w, tile_w - x)
    src_y2 = min(crop_h, tile_h - y)

    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return tile

    crop_part = crop[src_y1:src_y2, src_x1:src_x2]
    mask_part = mask[src_y1:src_y2, src_x1:src_x2]
    tile_part = tile[dst_y1:dst_y2, dst_x1:dst_x2]

    mask_bool = mask_part > 0
    tile_part[mask_bool] = crop_part[mask_bool]

    tile[dst_y1:dst_y2, dst_x1:dst_x2] = tile_part

    return tile


def create_synthetic_tiles(failed_crops):
    '''
    Create syntetic tiles for big crops and insert their, who does'n fit in
    save image, labels if tile with object
    :param failed_crops:
    :return: stats
    '''
    if not failed_crops:
        return {'created_tiles': 0, 'inserted': defaultdict(int)}

    print(f"Больших кропов не поместилось: {len(failed_crops)}")
    print(f"Создаём синтетические тайлы...\n")

    stats = {
        'created_tiles': 0,
        'inserted': defaultdict(int)
    }

    random.shuffle(failed_crops)

    tile_idx = 0
    crop_idx = 0

    while crop_idx < len(failed_crops) and tile_idx < MAX_SYNTHETIC_TILES:
        synthetic_tile = np.ones((SYNTHETIC_TILE_SIZE, SYNTHETIC_TILE_SIZE), dtype=np.uint8) * 255

        num_objects = random.randint(*OBJECTS_PER_SYNTHETIC_TILE)
        num_objects = min(num_objects, len(failed_crops) - crop_idx)

        if num_objects == 0:
            break

        placed_bboxes = []
        placed_objects = []

        for _ in range(num_objects):
            if crop_idx >= len(failed_crops):
                break

            class_id, crop, mask = failed_crops[crop_idx]
            crop_h, crop_w = crop.shape

            position_found = False

            for attempt in range(MAX_ATTEMPTS):
                if crop_w >= SYNTHETIC_TILE_SIZE or crop_h >= SYNTHETIC_TILE_SIZE:
                    x = (SYNTHETIC_TILE_SIZE - crop_w) // 2
                    y = (SYNTHETIC_TILE_SIZE - crop_h) // 2
                else:
                    x = random.randint(0, SYNTHETIC_TILE_SIZE - crop_w)
                    y = random.randint(0, SYNTHETIC_TILE_SIZE - crop_h)

                new_bbox = (
                    (x + crop_w / 2) / SYNTHETIC_TILE_SIZE,
                    (y + crop_h / 2) / SYNTHETIC_TILE_SIZE,
                    crop_w / SYNTHETIC_TILE_SIZE,
                    crop_h / SYNTHETIC_TILE_SIZE
                )

                overlap = any(calculate_iou(new_bbox, ex) > 0.05 for ex in placed_bboxes)

                if not overlap:
                    synthetic_tile = insert_object(synthetic_tile, crop, mask, x, y)
                    placed_bboxes.append(new_bbox)
                    placed_objects.append((class_id, new_bbox))
                    position_found = True
                    break

            if position_found:
                crop_idx += 1
            else:
                crop_idx += 1

        if len(placed_objects) > 0:
            tile_name = f"synthetic_{tile_idx:04d}"

            output_img = OUTPUT_IMAGES / f"{tile_name}.png"
            cv2.imwrite(str(output_img), synthetic_tile)

            output_label = OUTPUT_LABELS / f"{tile_name}.txt"
            with open(output_label, 'w') as f:
                for class_id, bbox in placed_objects:
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} "
                            f"{bbox[2]:.6f} {bbox[3]:.6f}\n")
                    stats['inserted'][class_id] += 1

            stats['created_tiles'] += 1
            tile_idx += 1

    print(f" Создано синтетических тайлов: {stats['created_tiles']}")
    print(f" Размещено объектов: {sum(stats['inserted'].values())}")

    return stats


def augment_dataset(objects):
    """
    Copy-Paste augmentation. if doesn't hit to create and save crop for synthetic tiles.
    :param objects:
    :return: stats {inserted,
                    failed,
                    by_type {full_visible, partial, occlusion}}
    """
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(list(TRAIN_IMAGES.glob("*.png")), desc="Coping"):
        shutil.copy(img_file, OUTPUT_IMAGES / img_file.name)
        label_file = TRAIN_LABELS / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, OUTPUT_LABELS / f"{img_file.stem}.txt")

    current_counts = defaultdict(int)
    for label_file in TRAIN_LABELS.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    current_counts[int(parts[0])] += 1

    stats = {
        'inserted': defaultdict(int),
        'failed': defaultdict(int),
        'by_type': {
            'full_visible': defaultdict(int),
            'partial': defaultdict(int),
            'occlusion': defaultdict(int)
        }
    }

    failed_crops = []
    tile_files = list(OUTPUT_IMAGES.glob("*.png"))

    for class_id in RARE_CLASSES:
        current = current_counts.get(class_id, 0)
        target = TARGET_COUNTS.get(class_id, 150)
        needed = target - current

        if needed <= 0:
            continue

        source_objects = objects.get(class_id, [])

        if len(source_objects) == 0:
            continue

        print(f"\n Класс {class_id}: нужно добавить {needed} объектов")
        pbar = tqdm(total=needed, desc=f"Paste")

        inserted = 0
        attempts = 0
        max_attempts = needed * 10
        consecutive_failures = 0

        while inserted < needed and attempts < max_attempts:
            attempts += 1

            img, bbox, source_name = random.choice(source_objects)

            crop, mask = extract_object_with_mask(img, bbox, padding=0.15)

            if crop is None or crop.shape[0] < MIN_OBJECT_SIZE or crop.shape[1] < MIN_OBJECT_SIZE:
                continue

            crop_aug, mask_aug = apply_augmentations(crop, mask)

            rand = random.random()

            if rand < INSERTION_RULES['full_visible']:
                insertion_type = 'full_visible'
            elif rand < INSERTION_RULES['full_visible'] + INSERTION_RULES['partial_80_90']:
                insertion_type = 'partial'
            else:
                insertion_type = 'occlusion'

            tile_file = random.choice(tile_files)
            tile = cv2.imread(str(tile_file), cv2.IMREAD_GRAYSCALE)

            if tile is None:
                continue

            forbidden_mask = load_forbidden_mask(tile_file.stem)

            label_file = OUTPUT_LABELS / f"{tile_file.stem}.txt"
            existing_bboxes = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            existing_bboxes.append(tuple(map(float, parts[1:5])))

            result = None

            if insertion_type == 'full_visible':
                result = find_position_full_visible(tile, crop_aug, forbidden_mask, existing_bboxes)
            elif insertion_type == 'partial':
                result = find_position_partial(tile, crop_aug, forbidden_mask, existing_bboxes)
            else:
                result = find_position_occlusion(tile, crop_aug, forbidden_mask, existing_bboxes)

            if result is None:
                stats['failed'][class_id] += 1
                consecutive_failures += 1

                if consecutive_failures >= 5:
                    failed_crops.append((class_id, crop_aug.copy(), mask_aug.copy()))
                    consecutive_failures = 0

                continue

            x, y, new_bbox = result

            tile = insert_object(tile, crop_aug, mask_aug, x, y)

            cv2.imwrite(str(tile_file), tile)

            with open(label_file, 'a') as f:
                f.write(f"{class_id} {new_bbox[0]:.6f} {new_bbox[1]:.6f} "
                        f"{new_bbox[2]:.6f} {new_bbox[3]:.6f}\n")

            stats['inserted'][class_id] += 1
            stats['by_type'][insertion_type][class_id] += 1

            inserted += 1
            consecutive_failures = 0
            pbar.update(1)

        pbar.close()

        if stats['failed'][class_id] > 0:
            print(f"   Неудачных попыток: {stats['failed'][class_id]}")

    if failed_crops:
        print(f"\n Обнаружено {len(failed_crops)} больших кропов, которые не поместились")
        print(f" Создаём синтетические тайлы для них...")

        synthetic_stats = create_synthetic_tiles(failed_crops)

        for class_id, count in synthetic_stats['inserted'].items():
            stats['inserted'][class_id] += count
            stats['by_type']['synthetic'] = stats['by_type'].get('synthetic', defaultdict(int))
            stats['by_type']['synthetic'][class_id] = count

        stats['synthetic_tiles_created'] = synthetic_stats['created_tiles']
    else:
        stats['synthetic_tiles_created'] = 0

    return stats


objects = collect_objects()

stats = augment_dataset(objects)

shutil.move(str(INPUT_TEST), str(OUTPUT_TEST))