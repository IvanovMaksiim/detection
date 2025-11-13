import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random

'''
Augmentation dataset by rotate 90, 180, 270
Variants: all 3, random 1 rotate, random 2 rotates
Only train
'''
random.seed(42)

BASE_DIR = Path(__file__).parent

TRAIN_IMAGES = BASE_DIR / 'dataset_two_stage'/ "train_augmented" / "images"
TRAIN_LABELS = BASE_DIR / 'dataset_two_stage'/ "train_augmented" / "labels"

TRAIN_IMAGES_OUT = BASE_DIR / 'dataset_two_stage'/ "train_augmented_rotated" / "images"
TRAIN_LABELS_OUT = BASE_DIR / 'dataset_two_stage'/ "train_augmented_rotated" / "labels"

ROTATION_ANGLES = [90, 180, 270]

# Modes
# 'all' - save all 3 rotate (90, 180, 270) + original = 4 img
#  'random_1' - save random 1 rotate + original = 2 img
#  'random_2' - save random 1 rotate + original = 3 img

AUGMENTATION_MODE = "all"

def select_angles_to_save(mode, available_angles):
    '''
    Chose angles for rotate
    :param mode: all, random_1, random_2
    :param available_angles: [90, 180, 270]
    :return: [angles]
    '''
    if mode == "all":
        return available_angles.copy()

    if mode == "random_1":
        return [random.choice(available_angles)] if available_angles else []

    if mode == "random_2":
        if len(available_angles) >= 2:
            return random.sample(available_angles, 2)
        else:
            return available_angles.copy()

def rotate_image(image, angle):
    '''
    Rotate image on angle
    :param image:
    :param angle:
    :return: rotate image
    '''

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_bbox_yolo(bbox, angle, img_width, img_height):
    '''
    Rotate YOLO bbox (normalized xywh) on angle
    :param bbox:
    :param angle:
    :param img_width:
    :param img_height:
    :return: rotated bbox yolo format
    '''
    x_center, y_center, width, height = bbox

    if angle == 90:
        # (x, y) → (1-y, x)
        new_x = 1 - y_center
        new_y = x_center
        new_width = height
        new_height = width

    elif angle == 180:
        # (x, y) → (1-x, 1-y)
        new_x = 1 - x_center
        new_y = 1 - y_center
        new_width = width
        new_height = height

    elif angle == 270:
        # (x, y) → (y, 1-x)
        new_x = y_center
        new_y = 1 - x_center
        new_width = height
        new_height = width

    return [new_x, new_y, new_width, new_height]

def augment_image_and_label(img_path, label_path, output_img_dir, output_label_dir,
                            angles, mode):
    '''
    Rotate 1 img with his label on list angles.
    Save new and old imgs and labels
    :param img_path:
    :param label_path:
    :param output_img_dir:
    :param output_label_dir:
    :param angles: [90, 180, 270]
    :param mode: ['all', 'random_1', 'random_2']
    :param img_index:
    :return: count save images
    '''
    img = cv2.imread(str(img_path))

    if img is None or not label_path.exists():
        return 0

    img_height, img_width = img.shape[:2]
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    bboxes = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            bboxes.append((class_id, bbox))

    img_name = img_path.stem

    shutil.copy(img_path, output_img_dir / f'{img_name}.png')
    shutil.copy(label_path, output_label_dir / f'{img_name}.txt')
    saved_count = 1

    angles_to_saved = select_angles_to_save(mode, angles)

    for angle in angles_to_saved:
        rotated_img = rotate_image(img, angle)

        rotated_bboxes = []

        for class_id, bbox in bboxes:
            rotated_bbox = rotate_bbox_yolo(bbox, angle, img_width, img_height)
            rotated_bboxes.append((class_id, rotated_bbox))

        output_img_path = output_img_dir / f'{img_name}_rot{angle}.png'
        cv2.imwrite(str(output_img_path), rotated_img)

        output_label_path = output_label_dir / f'{img_name}_rot{angle}.txt'
        with open(output_label_path, 'w', encoding='utf-8') as f:
            for class_id, bbox in rotated_bboxes:
                line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                f.write(line)

        saved_count += 1

    return saved_count



def process_train(images_dir, labels_dir, output_images_dir, output_labels_dir,
                 angles, mode):

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*png'))

    if not len(image_files):
        print(f'Not find img in {images_dir}')
        return

    mode_multipliers = {
        'all':4,
        'random_1':2,
        'random_2': 3
    }

    print(f'Mode: {mode}, expected images per original: {mode_multipliers.get(mode, "unknown")}')

    total_saved = 0
    for img_path in tqdm(image_files, desc='Augmentation'):
        label_path = labels_dir / f'{img_path.stem}.txt'
        saved = augment_image_and_label(
            img_path,
            label_path,
            output_images_dir,
            output_labels_dir,
            angles,
            mode
        )
        total_saved += saved

    result_images = len(list(output_images_dir.glob("*.png")))
    result_labels = len(list(output_labels_dir.glob("*.txt")))

    print(f'Save image {total_saved}')
    print(f'Images in folder {result_images}')
    print(f'Labels in folder {result_labels}')


process_train(
    TRAIN_IMAGES,
    TRAIN_LABELS,
    TRAIN_IMAGES_OUT,
    TRAIN_LABELS_OUT,
    ROTATION_ANGLES,
    AUGMENTATION_MODE
)