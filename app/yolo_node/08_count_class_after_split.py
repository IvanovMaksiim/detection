from pathlib import Path
from collections import defaultdict
import csv

"""
Analise of class after split
"""

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset_two_stage"
RARE_THRESHOLD = 50


def analysis_split(split_name):
    '''
    Analise of split
    :param split_name:
    :return: class_counts, class_sizes - diagonal of bbox
    '''
    labels_dir = DATASET_DIR / split_name / 'labels'
    class_counts = defaultdict(int)
    class_sizes = defaultdict(list)

    for label_file in labels_dir.glob('*txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                width = float(parts[3]) * 1024
                height = float(parts[4]) * 1024

                class_counts[class_id] += 1
                size = int((width ** 2 + height ** 2) ** 0.5)  # диагональ
                class_sizes[class_id].append(size)

    return class_counts, class_sizes


def save_csv(split_name, class_counts, class_sizes, class_names, rare_threshold):
    '''
    Save the result of analise to csv ["class_id", "class_name", "count", "min_size", "max_size", "rare"]
    :param split_name:
    :param class_counts:
    :param class_sizes:
    :param class_names:
    :param rare_threshold:
    '''
    path_csv = BASE_DIR/ 'statistic' / f'{split_name}_classes.csv'
    with open(path_csv, 'w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "count", "min_size", "max_size", "rare"])
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            sizes = class_sizes[class_id]
            min_size = min(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0
            rare = "yes" if count < rare_threshold else "no"
            class_name = class_names.get(class_id, f"unknown_{class_id}")
            writer.writerow([class_id, class_name, count, min_size, max_size, rare])


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
    32: "smotrowoe_steclo", 33: "datchik", 35: "output", 38: "strelka"
}

for split_name in ['train', 'val']:
    print(f"\n{split_name.upper()}:")

    class_counts, class_sizes = analysis_split(split_name)

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_id, count in sorted_classes:
        class_name = CLASS_NAMES.get(class_id, f'unknown {class_id}')
        sizes = class_sizes[class_id]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        print(f"Класс {class_id:2d} ({class_name:25s}): {count:5d} объектов, "
              f"размер {min_size:3d}-{max_size:4d}px")

    save_csv(split_name, class_counts, class_sizes, CLASS_NAMES, RARE_THRESHOLD)
