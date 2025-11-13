import os
import cv2
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR.parent / "../data" / "pages"
OUTPUT_DIR = os.path.join(INPUT_DIR, 'processed')

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]

for f_name in tqdm(files, desc='Processing_images', unit='img'):
    img_path = os.path.join(INPUT_DIR, f_name)
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_out = os.path.join(OUTPUT_DIR, f_name)
    cv2.imwrite(gray_out, gray)
