import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mnist_net import MNISTNet
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pytorch_lightning as pl
device = torch.device('cpu')
class MNISTLightningModel(pl.LightningModule):
    """PyTorch Lightning模型包装器"""

    def __init__(self):
        super().__init__()
        # 关注点1 模型结构
        self.model = MNISTNet().to(device)
        # 关注点2 损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 数据集方式1：online
        self.dataset = MNIST('/Users/cuijinlong/Documents/datasets/mnist_online',download=True, transform=transforms.ToTensor())

    def forward(self, x):
        return self.model(x)

    # 单批次数据训练逻辑
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.to(device))
        loss = self.criterion(y_hat, y.to(device))
        if batch_idx % 10 == 0:
            self.log('train_loss', loss, prog_bar=True)
        return loss

    # 关注点3 优化器
    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # 数据集方式1：online
    def train_dataloader(self):
        train_loader = DataLoader(self.dataset, batch_size=64, shuffle=True) # 由于这里被 dataset.py 取代了所以就不在这里写加载数据集了。
        return train_loader
if __name__ == '__main__':
    # 创建模型
    model = MNISTLightningModel()

    # 关注点5 训练逻辑
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu'
    )

    # 训练模型
    trainer.fit(model)

    # 测试模型
    trainer.test(model)