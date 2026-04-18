import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np


class SkinDataset(Dataset):
    def __init__(self, data_root, csv_path, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        # 1. 自动识别图像文件名列
        fname_col = 'image_name' if 'image_name' in self.df.columns else 'image_id'
        self.image_files = self.df[fname_col].values

        # 2. 自动识别标签列并进行数字化映射
        target_col = ''
        for col in ['class_name', 'dx', 'target', 'label']:
            if col in self.df.columns:
                target_col = col
                break

        raw_labels = self.df[target_col].values
        self.class_names = sorted(list(set(raw_labels)))
        self.label_map = {name: i for i, name in enumerate(self.class_names)}
        self.labels = np.array([self.label_map[name] for name in raw_labels])

        print(f"[*] 数据集加载成功: {len(self.df)} 张图片, {len(self.class_names)} 个类别")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = str(self.image_files[idx])
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fname += '.jpg'
        img_path = os.path.join(self.data_root, fname)

        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        return image, label


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label = torch.LongTensor(label)
        self.m_ind = []
        # 修正：确保只选取样本数足够的类别
        for i in range(int(label.max()) + 1):
            ind = torch.argwhere(label == i).reshape(-1)
            if len(ind) >= n_per:
                self.m_ind.append(ind)

        print(f"[*] 采样器就绪: 共有 {len(self.m_ind)} 个有效类别用于测试")

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            yield torch.stack(batch).reshape(-1)