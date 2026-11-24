"""Generate a minimal synthetic IU X-ray dataset for smoke testing.

This script creates a tiny synthetic dataset (16 studies: 8 train / 4 val /
4 test) that follows the exact data layout expected by the training pipeline:

    data/iu_xray/
    |-- images/                    # two synthetic frontal/lateral images per study
    |-- annotation.json            # R2Gen-style annotation file (train/val/test)
    `-- r2gen_split/id_label.csv   # study id -> 14 CheXpert-style binary labels

It allows verifying that the whole training pipeline (data loading,
tokenization, model forward/backward, sampling, NLG metric computation and
result saving) runs end-to-end without downloading the real IU X-ray dataset.

Usage:
    python scripts/make_iu_smoke_data.py
"""

import csv
import json
import os

import numpy as np
from PIL import Image

SEED = 42
IMAGE_SIZE = 256

# Split sizes (keep the vocabulary frequency >= threshold = 3 in train)
SPLITS = {'train': 8, 'val': 4, 'test': 4}

# Two report variants, each repeated 4 times in the train split so that every
# token frequency is >= 4 (>= the tokenizer threshold of 3)
REPORT_VARIANTS = [
    "the heart is normal in size. the lungs are clear. no acute abnormality.",
    "the cardiac silhouette is normal. the lungs are clear. no pleural effusion.",
]

# 14 CheXpert-style label column names
LABEL_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices',
]


def make_image(path, rng):
    """Create one synthetic grayscale 'X-ray' (smooth gradient + blobs + noise)."""
    x = np.linspace(0.0, 1.0, IMAGE_SIZE)
    base = np.outer(x, x) * 0.5 + 0.25
    img = base + rng.normal(0, 0.05, (IMAGE_SIZE, IMAGE_SIZE))
    # A few random roundish "lesion" blobs
    for _ in range(rng.randint(1, 4)):
        cy, cx = rng.randint(32, IMAGE_SIZE - 32, size=2)
        radius = rng.randint(8, 28)
        yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
        img[blob] += rng.uniform(0.1, 0.3)
    img = np.clip(img, 0.0, 1.0) * 255
    Image.fromarray(img.astype('uint8'), mode='L').convert('RGB').save(path)


def main():
    root = os.path.join('data', 'iu_xray')
    img_dir = os.path.join(root, 'images')
    split_dir = os.path.join(root, 'r2gen_split')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    rng = np.random.RandomState(SEED)

    annotation = {}
    label_rows = []
    study_id = 0

    for split, count in SPLITS.items():
        examples = []
        for i in range(count):
            study_id += 1
            # Study id format: CXR{N}_IM-XXXX-XXXX (pid is parsed as N)
            ex_id = f'CXR{study_id}_IM-{study_id:04d}-{study_id % 100:04d}'

            # Two views (frontal / lateral) per study
            image_paths = []
            for view in (0, 1):
                fname = f'{ex_id}_{view}.png'
                make_image(os.path.join(img_dir, fname), rng)
                image_paths.append(fname)

            report = REPORT_VARIANTS[(study_id - 1) % len(REPORT_VARIANTS)]
            examples.append({
                'id': ex_id,
                'image_path': image_paths,
                'report': report,
            })

            # Random 0/1 disease labels for this study
            label_row = {'id': study_id}
            for name in LABEL_NAMES:
                label_row[name] = int(rng.rand() < 0.2)
            label_rows.append(label_row)
        annotation[split] = examples

    # R2Gen-style annotation file
    ann_path = os.path.join(root, 'annotation.json')
    with open(ann_path, 'w') as f:
        json.dump(annotation, f, indent=2)

    # Study id -> label mapping (id column + 14 label columns)
    label_path = os.path.join(split_dir, 'id_label.csv')
    with open(label_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id'] + LABEL_NAMES)
        writer.writeheader()
        writer.writerows(label_rows)

    print(f'Synthetic IU X-ray smoke dataset generated under: {root}')
    print(f'  - {sum(SPLITS.values())} studies '
          f'({SPLITS["train"]} train / {SPLITS["val"]} val / {SPLITS["test"]} test)')
    print(f'  - {2 * sum(SPLITS.values())} images in {img_dir}')
    print(f'  - annotation: {ann_path}')
    print(f'  - labels:     {label_path}')
    print('Run the smoke test with:')
    print('  python main_dgmk.py --cfg config/iu_dgmk.yml --gpu 0 '
          '--set_cfgs epochs 1 batch_size 2 num_workers 0 expe_name smoke_test')


if __name__ == '__main__':
    main()
