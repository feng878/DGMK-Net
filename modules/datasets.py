import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        label_dict = {}

        # Fix for reading paths with non-ASCII characters on Windows
        data = pd.read_csv(label_path, engine='python')
        for index, row in data.iterrows():
            idx = row['id']
            # Compatible with older pandas versions
            label = row[1:].tolist() if hasattr(row[1:], 'tolist') else row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        pid = image_id.split('_')[0][3:]
        try:
            labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        except:
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.num_labels = args.num_labels  # store the number of labels
        # Optimization: convert the label table into a dict to speed up lookup
        self.label_dict = self._build_label_dict(args.label_path)

    def _build_label_dict(self, label_path):
        """Build a label dict to speed up lookups."""
        label_dict = {}
        df = pd.read_csv(label_path)

        # Convert the DataFrame into a dict: key is study_id, value is the label list
        for _, row in df.iterrows():
            study_id = int(row['study_id'])
            # CheXpert labels start at the 3rd column (after subject_id, study_id)
            labels = row.tolist()[2:]
            # Convert NaN to 0
            labels = [0.0 if pd.isna(x) else float(x) for x in labels]
            label_dict[study_id] = labels

        return label_dict


    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # The MIMIC-CXR label file is indexed by study_id.
        # image_id has the form 's50414267' and must be converted to the integer 50414267
        try:
            if image_id.startswith('s'):
                study_id_int = int(image_id[1:])
            else:
                # If it does not start with 's', try a direct match
                study_id_int = int(image_id)

            # Dict lookup instead of DataFrame filtering: a significant speedup
            if study_id_int in self.label_dict:
                labels = torch.tensor(self.label_dict[study_id_int], dtype=torch.float32)
            else:
                # Fall back to an all-zero label when no label is found
                labels = torch.tensor([0.0 for _ in range(self.num_labels)], dtype=torch.float32)
        except Exception as e:
            # On any error, fall back to an all-zero label
            labels = torch.tensor([0.0 for _ in range(self.num_labels)], dtype=torch.float32)
        
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample
