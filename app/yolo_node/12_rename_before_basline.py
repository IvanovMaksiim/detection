from pathlib import Path
from tqdm import tqdm
from collections import Counter
import yaml

"""
Rename in labels file 35 -> 34 (output), 38 -> 35(strelka). bc in 03_drop_class.py drop 3 class [34, 36, 37]
Create clean data.yaml
"""

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset_two_stage'

SPLITS_TO_REINDEX = {
    'train_augmented': DATASET_DIR / "train_augmented_rotated" / "labels",
    'val': DATASET_DIR / "val" / "labels",
    'test': DATASET_DIR / "test" / "labels"
}

CLASS_MAPPING = {
    35: 34,
    38: 35,
}

CREATE_BACKUP = False

def create_data_yaml():
    yaml_path = BASE_DIR / "data.yaml"

    data_yaml = {
        'path': './dataset_two_stage',
        'train': 'train_augmented_rotated/images',
        'val': 'val/images',
        'nc': 36,
        'names': {
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
            34: "output",
            35: "strelka"
        }
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\n data.yaml create: {yaml_path}")

def reindex_label_file(label_path, class_mapping):
    """
    Reindex class in 1 file

    Returns:
        (changed_count, total_lines)
    """
    if not label_path.exists():
        return 0, 0

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    changed = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        old_class = int(parts[0])

        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            parts[0] = str(new_class)
            changed += 1

        new_lines.append(' '.join(parts) + '\n')

    with open(label_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    return changed, len(lines)

def reindex_split(split_name, labels_dir):
        '''
        Reindex one split
        :param split_name:
        :param labels_dir:
        :return: stats of rename
        '''

        if not labels_dir.exists():
                return None

        # Статистика
        stats = {
                'total_files': 0,
                'files_changed': 0,
                'total_lines_changed': 0,
                'total_lines': 0,
                'class_changes': {old: 0 for old in CLASS_MAPPING.keys()}
                }

        label_files = list(labels_dir.glob("*.txt"))
        stats['total_files'] = len(label_files)

        for label_file in tqdm(label_files, desc=f"Reindex {split_name}"):
                changed, total = reindex_label_file(label_file, CLASS_MAPPING)

                if changed > 0:
                    stats['files_changed'] += 1
                    stats['total_lines_changed'] += changed

                stats['total_lines'] += total

        return stats

def verify_reindexing():
    '''
    Check result reindex
    :return: none
    '''

    all_classes = set()

    for split_name, labels_dir in SPLITS_TO_REINDEX.items():
        if not labels_dir.exists():
            continue

        classes = Counter()

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes[class_id] += 1
                        all_classes.add(class_id)

    if all_classes == set(range(36)):
        print("\n Congratulation all class reindex")
    else:
        missing = set(range(36)) - all_classes
        if missing:
            print(f"Missing class: {sorted(missing)}")
        extra = all_classes - set(range(36))
        if extra:
            print(f"Extra class: {sorted(extra)}")


all_stats = {}

for split_name, labels_dir in SPLITS_TO_REINDEX.items():
        stats = reindex_split(split_name, labels_dir)
        if stats:
            all_stats[split_name] = stats

verify_reindexing()

create_data_yaml()