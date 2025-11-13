from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

'''
Test on original images with SAHI
- Find better confidence thresholds
- Precision, Recall, F1, mAP
- Confusion matrix
- Per-class metrics
- Save COCO JSON
- Visualisation
'''
BASE_DIR = Path(__file__).parent

CONFIG = {
    'model_path': BASE_DIR / 'experiments' / 'baseline_augmented_m_2025-11-13' / 'weights'/ 'best.pt',
    'originals_dir': BASE_DIR / 'dataset_two_stage' / 'test',
    'results_dir': BASE_DIR / 'test_sahi_results',

    'slice_height': 1280,
    'slice_width': 1280,
    'overlap_ratio': 0.25,

    'confidence_thresholds': [0.8],
    'iou_threshold': 0.5,

    'device': 'cuda',
}

CLASS_NAMES = {
    0: "armatura_ruchn", 1: "klapan_obratn", 2: "regulator_ruchn",
    3: "armatura_electro", 4: "regulator_electro", 5: "drossel",
    6: "perehod", 7: "klapan_obratn_seroprivod", 8: "armatura_seroprivod",
    9: "regulator_seroprivod", 10: "armatura_membr_electro", 11: "nasos",
    12: "ventilaytor", 13: "predohran", 14: "condensatootvod",
    15: "rashodomernaya_shaiba", 16: "vodostruiniy_nasos", 17: "teploobmen",
    18: "zaglushka", 19: "gidrozatvor", 20: "bak", 21: "voronka",
    22: "filtr_meh", 23: "separator", 24: "kapleulov", 25: "celindr_turb",
    26: "redukcion_ustr", 27: "bistro_redukc_ustr", 28: "separator_paro",
    29: "dearator", 30: "silfonnii_kompensator", 31: "electronagrevat",
    32: "smotrowoe_steclo", 33: "datchik", 34: "output", 35: "strelka"
}

def load_ground_truth(images_dir, labels_dir):
    '''
    Download Ground Truth annotation
    :param images_dir:
    :param labels_dir:
    :return: {img_path, im_size, annotation}
    '''

    gt_data = {}
    class_counts = defaultdict(int)
    total_objects = 0

    for label_file in labels_dir.glob('*.txt'):
        img_name = label_file.stem
        img_path = images_dir / f'{img_name}.png'

        img = Image.open(img_path)
        img_w, img_h = img.size

        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h

                    x1, y1 = xc - w / 2, yc - h / 2
                    x2, y2 = xc + w / 2, yc + h / 2

                    annotations.append({
                        'class_id': cls_id,
                        'bbox': [x1, y1, x2, y2]
                    })

                    total_objects += 1
                    class_counts[cls_id] += 1
        gt_data[img_name] = {
            'path': img_path,
            'size': (img_w, img_h),
            'annotations': annotations
        }

    print(f"Images: {len(gt_data)}")
    print(f"Objects: {total_objects}")
    print(f"Classes: {len(class_counts)}")

    unknown_classes = set()
    for cls_id in class_counts.keys():
        if cls_id not in CLASS_NAMES:
            unknown_classes.add(cls_id)
    if unknown_classes:
        print(f'unknown classes: {sorted(unknown_classes)}')

    return gt_data, class_counts

def run_inference(model_path, gt_data, results_dir, slice_height, slice_width,
                  overlap_ratio, conf_threshold, device):
    '''
    Inference with SAHI and save predictions as YOLO format
    :param model_path:
    :param gt_data: {img_path, img_size, annotation}
    :param results_dir:
    :param slice_height:
    :param slice_width:
    :param overlap_ratio:
    :param conf_threshold:
    :param device:
    :return:
    '''
    model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=str(model_path),
            confidence_threshold=conf_threshold,
            device=device
        )

    all_preds = {}

    yolo_save_dir = results_dir / f"pred_yolo_conf_{conf_threshold}"
    yolo_save_dir.mkdir(parents=True, exist_ok=True)

    for img_name, gt_info in tqdm(gt_data.items()):
        img_path = gt_info['path']
        img_np = read_image(str(img_path))

        result = get_sliced_prediction(
            img_np,
            model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

        preds = []
        img_w, img_h = gt_info['size']

        for obj in result.object_prediction_list:
            if obj.score.value < conf_threshold:
                continue

            cls_name = obj.category.name
            cls_id = None
            for cid, cname in CLASS_NAMES.items():
                if cname == cls_name:
                    cls_id = cid
                    break

            bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = bbox

            preds.append({
                'class_id': cls_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(obj.score.value)
            })

        all_preds[img_name] = preds

        yolo_path = yolo_save_dir / f"{img_name}.txt"
        with open(yolo_path, "w") as f:
            for p in preds:
                cls_id = p['class_id']
                x1, y1, x2, y2 = p['bbox']
                conf = p['confidence']

                # Нормализация координат
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    print(f"\n YOLO predictions save in: {yolo_save_dir}")
    return all_preds

def calculate_iou(box1, box2):
    """Вычислить IoU между двумя bbox"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

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

def calculate_metrics(gt_data, all_predictions, class_counts, conf_threshold, iou_threshold):
    '''
    Calculate Precision, Recall, F1, TP/FP/FN all and per_class
    :param gt_data:
    :param all_predictions:
    :param class_counts:
    :param conf_threshold:
    :param iou_threshold:
    :return: {conf, precision, recall, f1, per_class, confusion}
                per_class = [{class_id, class_name, gt_count, precision, recall, f1,}]
    '''

    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    all_class_ids = set(CLASS_NAMES.keys())
    for gt_info in gt_data.values():
        for ann in gt_info['annotations']:
            all_class_ids.add(ann['class_id'])

    max_class_id = max(all_class_ids)
    n_classes = max_class_id + 1

    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    for img_name, gt_info in gt_data.items():
        gt_anns = gt_info['annotations']
        preds = all_predictions.get(img_name, [])

        gt_matched = [False] * len(gt_anns)
        pred_matched = [False] * len(preds)

        pred_indices = sorted(range(len(preds)),
                              key=lambda i: preds[i]['confidence'],
                              reverse=True)

        for pi in pred_indices:
            pred = preds[pi]
            best_iou = 0
            best_gt_idx = -1

            for gi, gt in enumerate(gt_anns):
                if gt_matched[gi]:
                    continue
                if gt['class_id'] != pred['class_id']:
                    continue

                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # TP
                pred_matched[pi] = True
                gt_matched[best_gt_idx] = True
                class_tp[pred['class_id']] += 1
                confusion[pred['class_id'], pred['class_id']] += 1
            else:
                # FP
                class_fp[pred['class_id']] += 1
                confusion[pred['class_id'], n_classes] += 1

        # FN
        for gi, gt in enumerate(gt_anns):
            if not gt_matched[gi]:
                class_fn[gt['class_id']] += 1
                confusion[n_classes, gt['class_id']] += 1
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  TP/FP/FN: {total_tp}/{total_fp}/{total_fn} \n")

    print(f"{'ID':<3} {'Class':<25} {'Images':>6} {'Inst.':>6} {'FP':>5} {'FN':>5} {'P':>8} {'R':>8} {'F1':>8}")

    per_class = []
    for cls_id in sorted(class_counts.keys()):
        tp = class_tp[cls_id]
        fp = class_fp[cls_id]
        fn = class_fn[cls_id]
        gt_count = class_counts[cls_id]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(
            f"{cls_id:>3} {CLASS_NAMES.get(cls_id, f'class_{cls_id}'):<25} {gt_count:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>8.3f} {rec:>8.3f} {f1_c:>8.3f}")


        per_class.append({
            'class_id': cls_id,
            'class_name': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
            'gt_count': gt_count,
            'precision': prec,
            'recall': rec,
            'f1': f1_c
        })

    return {
        'conf': conf_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class': per_class,
        'confusion': confusion
    }

def save_confusion_matrix(confusion, conf_threshold, results_dir):
    '''
    Saving confusion matrix as img real numbers and normalize
    :param confusion:
    :param conf_threshold:
    :param results_dir:
    :return: None
    '''
    plt.figure(figsize=(20, 18))

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums,
                               where=row_sums != 0,
                               out=np.zeros_like(confusion, dtype=float))

    n_classes = confusion.shape[0] - 1
    labels = []
    for i in range(n_classes):
        if i in CLASS_NAMES:
            labels.append(CLASS_NAMES[i])
        else:
            labels.append(f'class_{i}')
    labels.append('background')

    sns.heatmap(confusion_norm, annot=False, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized Count'})

    plt.title(f'Confusion Matrix (conf={conf_threshold})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()

    save_path = results_dir / f'confusion_matrix_conf{conf_threshold}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Confusion matrix saved: {save_path}")

def rename_yolo_predictions(results_dir, conf_threshold):
    '''
    Reindex class and remove confidence in labels
    34→35 (output), 35→38 (strelka)
    :param results_dir:
    :param conf_threshold:
    :return: None
    '''
    input_dir = results_dir / f"pred_yolo_conf_{conf_threshold}"
    output_dir = input_dir.parent / f"{input_dir.name}_renamed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # БЕЗОПАСНЫЙ маппинг
    old_to_new = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
        17: 17, 18: 18, 19: 19, 20: 20, 21: 21,
        22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
        27: 27, 28: 28, 29: 29, 30: 30, 31: 31,
        32: 32, 33: 33,
        34: 35,  # output
        35: 38  # strelka
    }


    for txt_path in sorted(input_dir.glob("*.txt")):
        new_lines = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    old_id = int(parts[0])
                except ValueError:
                    continue


                new_id = old_to_new.get(old_id, old_id)
                parts[0] = str(new_id)
                parts = parts[:5]  # Убрать confidence
                new_lines.append(" ".join(parts))

        out_path = output_dir / txt_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

    print(f"\n Результат сохранён в: {output_dir}")

def compare_results(all_results, results_dir):

    df = pd.DataFrame([{
        'conf': r['conf'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1']
    } for r in all_results])

    print(df.to_string(index=False))

    best_idx = df['f1'].idxmax()
    best = df.iloc[best_idx]

    print(f"\n Best: conf={best['conf']}")
    print(f"   F1: {best['f1']:.4f}")

    df.to_csv(results_dir / 'comparison.csv', index=False)
    print(f"\n Сохранено: {results_dir}/comparison.csv")

results_dir = Path(CONFIG['results_dir'])
results_dir.mkdir(parents=True, exist_ok=True)

images_dir = Path(CONFIG['originals_dir']) / 'images'
labels_dir = Path(CONFIG['originals_dir']) / 'labels'

gt_data, class_counts = load_ground_truth(images_dir, labels_dir)

all_results = []

for conf in CONFIG['confidence_thresholds']:
    preds = run_inference(
        model_path=CONFIG['model_path'],
        gt_data=gt_data,
        results_dir=results_dir,
        slice_height=CONFIG['slice_height'],
        slice_width=CONFIG['slice_width'],
        overlap_ratio=CONFIG['overlap_ratio'],
        conf_threshold=conf,
        device=CONFIG['device']
    )

    metrics = calculate_metrics(
        gt_data=gt_data,
        all_predictions=preds,
        class_counts=class_counts,
        conf_threshold=conf,
        iou_threshold=CONFIG['iou_threshold']
    )
    all_results.append(metrics)

    save_confusion_matrix(metrics['confusion'], conf, results_dir)

    rename_yolo_predictions(results_dir, conf)

compare_results(all_results, results_dir)
