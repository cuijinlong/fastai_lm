from torch import nn
import torch.optim as optim
from mnist_net import MNISTNet
from example.dataset import MNISTImageDataModule
import pytorch_lightning as pl

class MNISTLightningModel(pl.LightningModule):
    """PyTorch Lightning模型包装器"""

    def __init__(self):
        super().__init__()
        # 关注点1 模型结构
        self.model = MNISTNet()
        # 关注点2 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    # 单批次数据训练逻辑
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if batch_idx % 10 == 0:
            self.log('train_loss', loss, prog_bar=True)
        return loss

    # 关注点3 优化器
    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # 数据集方式1：online
    # def train_dataloader(self):
    #     # 由于这里被 dataset.py 取代了所以就不在这里写加载数据集了。
    #     return None


if __name__ == '__main__':
    # 自定义数据集
    data_module = MNISTImageDataModule(
        data_dir='/Users/cuijinlong/Documents/datasets/mnist_custom',
        batch_size=64,
        num_workers=0
    )
    # 创建模型
    model = MNISTLightningModel()

    # 关注点5 训练逻辑
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu'
    )

    # 训练模型
    trainer.fit(model, datamodule=data_module)

    # 测试模型
    trainer.test(model, datamodule=data_module)
