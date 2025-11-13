import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


"""
Тайлинг trainval с одновременным созданием forbidden масок (узлы и трубы - места куда нельзя copy-paste). 
"""
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / 'raw_data_split_two_stage'/ 'trainval'
OUTPUT_DIR = BASE_DIR / 'dataset_two_stage' / 'trainval_tiles'

TRUBA_MASKS_DIR = BASE_DIR.parent / '../mask_truba'
ANNOTATION_MASKS_DIR = BASE_DIR.parent / '../mask_annotation'

TILE_SIZE = 1280
OVERLAP = 0.25
MIN_AREA_RATIO = 0.6
KEEP_EMPTY_RATIO = 0.1

input_images = INPUT_DIR / 'images'
input_labels = INPUT_DIR / 'labels'

output_images = OUTPUT_DIR / 'images'
output_labels = OUTPUT_DIR / 'labels'
output_masks = OUTPUT_DIR / 'forbidden_masks'

output_images.mkdir(exist_ok=True, parents=True)
output_labels.mkdir(exist_ok=True, parents=True)
output_masks.mkdir(exist_ok=True, parents=True)

stats = {
    'processed_images': 0,
    'images_with_masks': 0,
    'tiles_with_objects': 0,
    'empty_tiles': 0,
    'total_objects': 0,
    'cropped': 0,
    'mask_forbidden_ratios': []
}
# Function for mask
def find_mask_file(scheme_name, mask_dir, suffix=''):
    '''
    Find mask's file for scheme
    :param scheme_name:
    :param mask_dir: start path
    :param suffix: for mask truba
    :return: full path or None
    '''
    variants = [
        mask_dir / f'{scheme_name}{suffix}.png',
        mask_dir / f'{scheme_name}{suffix}.jpg'
    ]
    for path in variants:
        if path.exists():
            return path
    return None

def load_and_combine_masks(scheme_name):
    """
    Загрузить и объединить маски трубы и аннотаций для схемы
    :param scheme_name:
    :return: forbidden_mask(0, 255) + dilate kernel(7.7) or None
    """

    truba_mask_path = find_mask_file(scheme_name, TRUBA_MASKS_DIR, '_mask')

    truba_mask = cv2.imread(str(truba_mask_path), cv2.IMREAD_GRAYSCALE)
    if truba_mask is None:
        return None

    forbidden = (truba_mask > 0).astype(np.uint8)

    annotation_mask_path = find_mask_file(scheme_name, ANNOTATION_MASKS_DIR)

    if annotation_mask_path is not None:
        annotation_mask = cv2.imread(str(annotation_mask_path), cv2.IMREAD_GRAYSCALE)

        if annotation_mask is not None:
            forbidden = np.logical_or(forbidden, annotation_mask > 0).astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    forbidden = cv2.dilate(forbidden*255, kernel)

    return forbidden


#  Function for tiles
def parse_yolo_label(label_path, img_width, img_height):
    """
    Parsing YOLO labels file
    :param label_path:
    :param img_width:
    :param img_height:
    :return: list of dictionary with data about each bbox [{class_id, x_center,
                                            y_center, width, height}]
    """
    if not label_path.exists():
        return []

    objects = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])*img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            objects.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    return objects
def bbox_intersection_area(obj, tile_x1, tile_y1, tile_x2, tile_y2):
    """Area intersection of object with tile"""
    obj_x1 = obj['x_center'] - obj['width'] / 2
    obj_y1 = obj['y_center'] - obj['height'] / 2
    obj_x2 = obj['x_center'] + obj['width'] / 2
    obj_y2 = obj['y_center'] + obj['height'] / 2

    inter_x1 = max(obj_x1, tile_x1)
    inter_y1 = max(obj_y1, tile_y1)
    inter_x2 = min(obj_x2, tile_x2)
    inter_y2 = min(obj_y2, tile_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0, 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    obj_area = obj['width'] * obj['height']

    return inter_area, obj_area

def clip_bbox_to_tile(obj, tile_x1, tile_y1, tile_size):
    '''
    Crop bbox of object along the borders of tile
    :param obj:
    :param tile_x1:
    :param tile_y1:
    :param tile_size:
    :return: new object [{class_id, x_center, y_center, width, height}] or None
    '''
    obj_x1 = obj['x_center'] - obj['width'] / 2
    obj_x2 = obj['x_center'] + obj['width'] / 2
    obj_y1 = obj['y_center'] - obj['height'] / 2
    obj_y2 = obj['y_center'] + obj['height'] / 2

    tile_x2 = tile_x1 + tile_size
    tile_y2 = tile_y1 + tile_size

    clipped_x1 = max(tile_x1, obj_x1)
    clipped_x2 = min(tile_x2, obj_x2)
    clipped_y1 = max(tile_y1, obj_y1)
    clipped_y2 = min(tile_y2, obj_y2)

    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return None

    new_x1 = clipped_x1 - tile_x1
    new_x2 = clipped_x2 - tile_x1
    new_y1 = clipped_y1 - tile_y1
    new_y2 = clipped_y2 - tile_y1

    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1
    new_x_center = (new_x1 + new_x2) / 2 / tile_size
    new_y_center = (new_y1 + new_y2) / 2 / tile_size
    new_width_norm = new_width / tile_size
    new_height_norm = new_height / tile_size

    return {
        'class_id': obj['class_id'],
        'x_center': new_x_center,
        'y_center': new_y_center,
        'width': new_width_norm,
        'height': new_height_norm
    }

def tile_image_with_mask(img_path, label_path, full_mask,
                         output_images, output_labels, output_masks, stats):
    """
    Tiling img + full_mask together

    :param img_path:
    :param label_path:
    :param full_mask:
    :param output_images:
    :param output_labels:
    :param output_masks:
    :param stats:
    :return: None
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Download failed {img_path.name} ')

    img_height, img_width = img.shape
    objects = parse_yolo_label(label_path, img_width,img_height)

    stride = int(TILE_SIZE*(1-OVERLAP))
    tile_idx = 0
    empty_tiles = []

    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            tile_x1 = x
            tile_y1 = y
            tile_x2 = min(x + TILE_SIZE, img_width)
            tile_y2 = min(y + TILE_SIZE, img_height)

            tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]

            if full_mask is not None:
                tile_mask = full_mask[tile_y1:tile_y2, tile_x1:tile_x2]
            else:
                tile_mask = None


            if (tile_x2 - tile_x1) != TILE_SIZE or (tile_y2 - tile_y1) != TILE_SIZE:
                h, w = tile_img.shape[:2]
                pad_bottom = TILE_SIZE - h
                pad_right = TILE_SIZE - w

                tile_img = cv2.copyMakeBorder(
                                    tile_img,
                                    top=0,
                                    bottom=pad_bottom,
                                    left=0,
                                    right=pad_right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=255
                                )

                if tile_mask is not None:
                    tile_mask = cv2.copyMakeBorder(
                        tile_mask,
                        top=0,
                        bottom=pad_bottom,
                        left=0,
                        right=pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=0
                    )

            tile_objects = []
            for obj in objects:
                inter_area, obj_area = bbox_intersection_area(obj,
                                                              tile_x1,
                                                              tile_y1,
                                                              tile_x2,
                                                              tile_y2)
                if inter_area == 0:
                    continue

                visible_ratio = inter_area / obj_area
                if visible_ratio < MIN_AREA_RATIO:
                    stats['cropped'] += 1
                    continue

                clipped = clip_bbox_to_tile(obj,
                                            tile_x1,
                                            tile_y1,TILE_SIZE)
                if clipped:
                    tile_objects.append(clipped)

            tile_name = f'{img_path.stem}_tile_{tile_idx:04d}'

            if len(tile_objects) == 0:
                empty_tiles.append((tile_name, tile_img, tile_mask))

            else:
                cv2.imwrite(str(output_images / f'{tile_name}.png'), tile_img)

                with open(output_labels / f'{tile_name}.txt', 'w', encoding='utf-8') as f:
                    for obj in tile_objects:
                        f.write(f"{obj['class_id']} {obj['x_center']:.6f} {obj['y_center']:.6f} "
                                f"{obj['width']:.6f} {obj['height']:.6f}\n")

                if tile_mask is not None:
                    cv2.imwrite(str(output_masks / f'{tile_name}_forbidden.png'), tile_mask)

                    forbidden_ratio = (tile_mask > 0).sum() / tile_mask.size
                    stats['mask_forbidden_ratios'].append(forbidden_ratio)

                else:
                    empty_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                    cv2.imwrite(str(output_masks / f'{tile_name}_forbidden.png'), empty_mask)

                stats['tiles_with_objects'] += 1
                stats['total_objects'] += len(tile_objects)

            tile_idx += 1

    num_empty_to_keep = int(len(empty_tiles) * KEEP_EMPTY_RATIO)
    if num_empty_to_keep > 0:
        np.random.shuffle(empty_tiles)

        for tile_name, tile_img, tile_mask in empty_tiles[:num_empty_to_keep]:

            cv2.imwrite(str(output_images / f'{tile_name}.png'), tile_img)

            (output_labels / f'{tile_name}.txt').touch()

            if tile_mask is not None:
                cv2.imwrite(str(output_masks / f'{tile_name}_forbidden.png'), tile_mask)
            else:
                empty_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                cv2.imwrite(str(output_masks / f'{tile_name}_forbidden.png'), empty_mask)

    stats['processed_images'] += 1
    if full_mask is not None:
        stats['images_with_masks'] += 1



image_files = sorted(input_images.glob('*.png'))
print(f"\nСхем в TrainVal: {len(image_files)}")

for img_path in tqdm(image_files, desc='Tiling TrainVal + Masks', unit='img'):
    scheme_name = img_path.stem
    labels_path = input_labels / f'{scheme_name}.txt'

    full_mask = load_and_combine_masks(scheme_name)

    tile_image_with_mask(img_path,
                         labels_path,
                         full_mask,
                         output_images,
                         output_labels,
                         output_masks,
                         stats)

print(f"  Схем: {stats['processed_images']}")
print(f"  Схем с масками: {stats['images_with_masks']}")
print(f"  Тайлов с объектами: {stats['tiles_with_objects']}")
print(f"  Пустых тайлов: {stats['empty_tiles']}")
print(f"  Всего тайлов: {stats['tiles_with_objects'] + stats['empty_tiles']}")
print(f"  Объектов: {stats['total_objects']}")
print(f"  Обрезано (< 60%): {stats['cropped']}")

if stats['mask_forbidden_ratios']:
    avg_forbidden = np.mean(stats['mask_forbidden_ratios']) * 100
    print(f"  Средний % запрещённой площади: {avg_forbidden:.1f}%")
