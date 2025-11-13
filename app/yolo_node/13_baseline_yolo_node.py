from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import multiprocessing

'''
Baseline learning YOLOv8 on augmetation dataset (rotate, cpa)
'''

MODEL_NAME = 'yolov8m.pt'

EPOCHS = 100
BATCH_SIZE = 4
IMG_SIZE = 1280
PATIENCE = 25

BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / 'data.yaml'
PROJECT_NAME = BASE_DIR / 'experiments'
RUN_NAME = f"baseline_augmented_m_{datetime.now():%Y-%m-%d}"

AUGMENT = False

def main():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,

        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,


        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        augment=AUGMENT,
        degrees=0,
        translate=0,  # 0.1
        scale=0,  # 0.2
        flipud=0,
        fliplr=0,  # 0.5
        mosaic=0,  # 0.5
        mixup=0,  # 0.1

        box=7.5,
        cls=0.5,
        dfl=1.5,

        save_period=10,
        verbose=True,
        plots=True,
        device=0,
    )

    metrics = model.val()

    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print(f" Results: {Path(PROJECT_NAME) / RUN_NAME}")

if __name__ == "__main__":
        multiprocessing.freeze_support()
        main()