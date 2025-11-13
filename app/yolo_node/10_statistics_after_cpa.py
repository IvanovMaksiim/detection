from pathlib import Path
from collections import Counter
import csv
'''
Analise after Copy-Paste Augmentation. 
Count final class in train, train_augmentation, val, test
'''
CLASS_NAMES = {
    0: "armatura_ruchn",
    1: "klapan_obratn",
    2: "regulator_ruchn",
    3: "armatura_electro",
    4: "regulator_electro",
    5: "drossel",
    6: "perehod",
    7: "klapan_obratn_seroprivod",
    8: "armatura_seroprivod",
    9: "regulator_seroprivod",
    10: "armatura_membr_electro",
    11: "nasos",
    12: "ventilaytor",
    13: "predohran",
    14: "condensatootvod",
    15: "rashodomernaya_shaiba",
    16: "vodostruiniy_nasos",
    17: "teploobmen",
    18: "zaglushka",
    19: "gidrozatvor",
    20: "bak",
    21: "voronka",
    22: "filtr_meh",
    23: "separator",
    24: "kapleulov",
    25: "celindr_turb",
    26: "redukcion_ustr",
    27: "bistro_redukc_ustr",
    28: "separator_paro",
    29: "dearator",
    30: "silfonnii_kompensator",
    31: "electronagrevat",
    32: "smotrowoe_steclo",
    33: "datchik",
    34: "annotation",
    35: "output",
    36: "truba",
    37: "unknow",
    38: "strelka"
}

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset_two_stage'

def analyze_split(split_name, labels_dir):
    '''
    Counter class_id in yolo.txt
    :param split_name:
    :param labels_dir:
    :return: class_counts, total_objects
    '''
    if not labels_dir.exists():
        return None

    class_counts = Counter()
    total_objects = 0

    for label_file in labels_dir.glob('*txt'):
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    return class_counts, total_objects

splits = {
    'train': DATASET_DIR / "train" / "labels",
    'train_augmented': DATASET_DIR / "train_augmented" / "labels",
    'val': DATASET_DIR / "val" / "labels",
    'test': DATASET_DIR / "test" / "labels"
}

all_classes_found = set()

for split_name, labels_dir in splits.items():
    print(f" {split_name.upper()}")

    result = analyze_split(split_name, labels_dir)

    if result is None:
        print(f'Folder {labels_dir} not find ')
        continue

    class_counts, total_objects = result

    csv_rows = []

    for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        all_classes_found.add(class_id)
        name = CLASS_NAMES.get(class_id, f'unknown_class {class_id}')
        pct = count/total_objects * 100

        csv_rows.append({
                'split': split_name,
                'id': class_id,
                'name': name,
                'count': count,
                'percent': round(pct, 2)
            })

        print(f"{class_id:<4} {name:<35} {count:>8} {pct:>6.1f}% ")

    csv_file = BASE_DIR / 'statistic' / f"class_diagnosis_{split_name}.csv"

    with open(csv_file, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'id', 'name', 'count', 'percent'])
        writer.writeheader()
        writer.writerows(csv_rows)

print(f"All unique class at dataset: {len(all_classes_found)}")
print(f"Class in obj.txt: {len(CLASS_NAMES)}")

missing = set(CLASS_NAMES.keys()) - all_classes_found
if missing:
    print(f'Classes who doesnt exist in dataset')
    for class_id in sorted(missing):
        print(f'{class_id}: {CLASS_NAMES[class_id]}')

extra = all_classes_found - set(CLASS_NAMES.keys())
if extra:
    print(f'Classes who doesnt exist in obj.txt')
    for class_id in sorted(extra):
        print(f'{class_id}')


