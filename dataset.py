import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os

class MNISTImageDataset(Dataset):
    """自定义MNIST图像数据集"""
    
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径
            image_dir (string): 图像目录的路径
            transform (callable, optional): 可选的图像变换
        """
        self.labels_df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # 如果没有提供transform，使用默认的
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # 获取文件名和标签
        img_name = self.labels_df.iloc[idx]['filename']
        label = self.labels_df.iloc[idx]['label']
        
        # 加载图像
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MNISTImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning数据模块"""
    
    def __init__(self, data_dir='/Users/cuijinlong/Documents/datasets/mnist', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def setup(self, stage=None):
        # 训练集
        self.train_dataset = MNISTImageDataset(
            csv_file=os.path.join(self.data_dir, 'train_labels.csv'),
            image_dir=os.path.join(self.data_dir, 'train'),
            transform=self.transform
        )
        
        # 测试集
        self.test_dataset = MNISTImageDataset(
            csv_file=os.path.join(self.data_dir, 'test_labels.csv'),
            image_dir=os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )
        
        # 从训练集中划分验证集 (80% 训练, 20% 验证)
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)