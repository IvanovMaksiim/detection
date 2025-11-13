from pathlib import Path
from collections import defaultdict
import random
import shutil
from tqdm import tqdm

"""
Двухэтапный split:
Этап 1: Схемы → Test (20%) / TrainVal (80%)
Этап 2: TrainVal → тайлинг → Train / Val (на уровне тайлов)
"""
random.seed(42)

BASE_DIR = Path(__file__).parent
INPUT_IMAGES = BASE_DIR / 'raw_data_final' / 'images'
INPUT_LABELS = BASE_DIR / 'raw_data_final' / 'labels'
OUTPUT_DIR = BASE_DIR / 'raw_data_split_two_stage'

TEST_RATIO = 0.2
RARE_CLASSES = [7, 9, 10, 12, 16, 18, 19, 23, 26, 27, 28, 30, 31]


def get_scheme_classes(label_path):
    classes = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes


"""
ЭТАП 1: Разделить схемы на Test / TrainVal

Правило для редких классов:
- Если <=3 схемы: ВСЕ в TrainVal
- Если 4-5 схем: минимум 1 в Test, остальные в TrainVal
- Обычные: 20% в Test, 80% в TrainVal
"""


label_files = sorted(INPUT_LABELS.glob('*.txt'))

scheme_to_classes = {}

for label_path in label_files:
    scheme_name = label_path.stem
    classes = get_scheme_classes(label_path)
    scheme_to_classes[scheme_name] = classes

class_to_scheme = defaultdict(list)

for scheme, classes in scheme_to_classes.items():
    for cls in classes:
        class_to_scheme[cls].append(scheme)

all_schemes = list(scheme_to_classes.keys())
total = len(all_schemes)

scheme_assignment = {}

for cls in tqdm(RARE_CLASSES, desc='Rare_classes', unit='class'):
    schemes = class_to_scheme.get(cls, [])
    n_schemes = len(schemes)

    if not n_schemes:
        continue

    schemes_copy = schemes.copy()
    random.shuffle(schemes_copy)

    if n_schemes <= 3:
        for scheme in schemes_copy:
            if scheme not in scheme_assignment:
                scheme_assignment[scheme] = 'trainval'
        print(f"Класс {cls:2d} ({n_schemes} схем): ВСЕ в TrainVal")

    else:
        for n_scheme in range(len(schemes_copy)):
            if scheme_assignment.get(schemes_copy[n_scheme]) != 'trainval':
                scheme_assignment[schemes_copy[n_scheme]] = 'test'
                break

        for scheme in schemes_copy:
            if scheme not in scheme_assignment:
                scheme_assignment[scheme] = 'trainval'

        test_count = sum(1 for s in schemes_copy if scheme_assignment[s] == 'test')
        trainval_count = n_schemes - test_count
        print(f"Класс {cls:2d} ({n_schemes} схем): {trainval_count} TrainVal, {test_count} Test")

unassigned = [s for s in all_schemes if s not in scheme_assignment]
print(f"\nОставшихся схем: {len(unassigned)}")

random.shuffle(unassigned)

target_test = int(total*TEST_RATIO)
current_test = sum(1 for s in scheme_assignment.values() if s == 'test')

needed_test = max(0, target_test - current_test)

for i, scheme in enumerate(unassigned):
    if i < needed_test:
        scheme_assignment[scheme] = 'test'
    else:
        scheme_assignment[scheme] = 'trainval'

test_scheme = [s for s, split in scheme_assignment.items() if split == 'test']
trainval_scheme = [s for s, split in scheme_assignment.items() if split == 'trainval']

assert len(set(test_scheme) & set(trainval_scheme)) == 0
print(f"Нет пересечений")

trainval_classes = set()
test_classes = set()

for scheme in trainval_scheme:
    trainval_classes.update(scheme_to_classes[scheme])
for scheme in test_scheme:
    test_classes.update(scheme_to_classes[scheme])

print(f"Классов в TrainVal: {len(trainval_classes)}")
print(f"Классов в Test:     {len(test_classes)}")

missing_in_trainval = []
for cls in RARE_CLASSES:
    if cls not in trainval_classes:
        missing_in_trainval.append(cls)
        print(f" Класс {cls}: НЕТ в TrainVal!")

if not missing_in_trainval:
    print(f"Все редкие классы в TrainVal")

for split, scheme in [('test', test_scheme), ('trainval', trainval_scheme)]:
    split_dir = OUTPUT_DIR / split
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'

    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)

    for scheme_name in scheme:
        src_img = INPUT_IMAGES / f'{scheme_name}.png'
        if src_img.exists():
            shutil.copy(src_img, images_dir/f'{scheme_name}.png')

        src_labels = INPUT_LABELS / f'{scheme_name}.txt'
        if src_labels.exists():
            shutil.copy(src_labels, labels_dir / f'{scheme_name}.txt')
    print(f"{split}: скопировано {len(scheme)} схем")


