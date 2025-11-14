# dataset.py 改进
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os
from typing import Optional


class MNISTImageDataset(Dataset):
    """自定义MNIST图像数据集"""

    def __init__(self, csv_file: str, image_dir: str, transform: Optional[callable] = None):
        """
        Args:
            csv_file (string): CSV文件的路径
            image_dir (string): 图像目录的路径
            transform (callable, optional): 可选的图像变换
        """
        self.labels_df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transform

        # 验证数据完整性
        self._validate_dataset()

    def _validate_dataset(self):
        """验证数据集完整性"""
        for idx in range(len(self.labels_df)):
            img_name = self.labels_df.iloc[idx]['filename']
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 获取文件名和标签
        img_name = self.labels_df.iloc[idx]['filename']
        label = self.labels_df.iloc[idx]['label']

        # 加载图像
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('L')  # 转换为灰度图像
        except Exception as e:
            raise RuntimeError(f"无法加载图像 {img_path}: {str(e)}")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


class MNISTImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning数据模块"""

    def __init__(self,
                 data_dir: str = '/Users/cuijinlong/Documents/datasets/mnist_custom',
                 batch_size: int = 64,
                 num_workers: int = 0,
                 validation_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split

        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 数据增强变换（仅用于训练）
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage: Optional[str] = None):
        # 训练集
        train_dataset = MNISTImageDataset(
            csv_file=os.path.join(self.data_dir, 'train_labels.csv'),
            image_dir=os.path.join(self.data_dir, 'train'),
            transform=self.train_transform  # 训练时使用数据增强
        )

        # 测试集
        self.test_dataset = MNISTImageDataset(
            csv_file=os.path.join(self.data_dir, 'test_labels.csv'),
            image_dir=os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )

        # 从训练集中划分验证集
        train_size = int((1 - self.validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size

        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # 验证集不使用数据增强
        self.mnist_val.dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )