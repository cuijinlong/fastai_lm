"""
基于PyTorch Lightning的手写数字识别训练脚本
作者: [您的姓名]
日期: [当前日期]
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import torch.nn as nn
import math
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from dataset import MNISTImageDataModule


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为输入序列添加位置信息
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: 嵌入维度
            dropout: dropout比率
            max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 使用正弦和余弦函数生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 注册为缓冲区，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]

        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ImageTransformerClassifier(pl.LightningModule):
    """
    基于Transformer的图像分类器
    """

    def __init__(
            self,
            num_classes: int = 10,
            d_model: int = 128,
            nhead: int = 8, # d_model(128)维被平均分成8份，每份 128/8 = 16维，8个头的输出拼接回128维
            num_layers: int = 3,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout_rate: float = 0.1
    ):
        """
        Args:
            num_classes: 分类类别数
            d_model: Transformer隐藏层维度
            nhead: 多头注意力头数
            num_layers: Transformer编码器层数
            learning_rate: 学习率
            weight_decay: 权重衰减
            dropout_rate: dropout比率
        """
        super().__init__()
        self.save_hyperparameters()  # 这等价于手动执行：
        # self.hparams = {
        #     'learning_rate': learning_rate,
        #     'num_layers': num_layers,
        #     'hidden_size': hidden_size
        # } 然后你可以通过 self.hparams 访问这些参数，例如：
        # self.learning_rate = self.hparams.learning_rate

        # 图像到序列的投影 (28x28 = 784 patches)
        self.patch_embed = nn.Linear(784, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=False  # 使用 [seq_len, batch, features] 格式
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # 损失函数和指标
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像张量 [batch_size, 1, 28, 28]

        Returns:
            分类logits [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # 展平图像 [batch_size, 784]
        x = x.view(batch_size, -1)

        # 添加序列维度并投影 [batch_size, 1, d_model]
        x = x.unsqueeze(1)
        x = self.patch_embed(x)

        # 调整维度并添加位置编码 [seq_len=1, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)

        # Transformer编码 [seq_len, batch_size, d_model]
        x = self.transformer(x)

        # 全局平均池化并分类 [batch_size, num_classes]
        x = x.mean(dim=0)  # 沿序列维度平均
        return self.classifier(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        验证步骤
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        测试步骤
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class MNISTPredictor:
    """
    MNIST模型预测器，提供便捷的预测接口
    """

    def __init__(self, model_path: Optional[str] = None, model: Optional[pl.LightningModule] = None):
        """
        Args:
            model_path: 模型文件路径
            model: 已经加载的模型实例
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self.load_model(model_path)
        else:
            raise ValueError("必须提供model_path或model参数")

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    @staticmethod
    def load_model(model_path: str) -> ImageTransformerClassifier:
        """
        加载训练好的模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载的模型实例
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建模型实例并加载权重
        model = ImageTransformerClassifier()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        return model

    def predict(self, image_path: str) -> Tuple[int, float, np.ndarray]:
        """
        预测单张图像

        Args:
            image_path: 图像文件路径

        Returns:
            predicted_label: 预测标签
            confidence: 置信度
            probabilities: 所有类别的概率分布
        """
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {str(e)}")

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度

        # 预测
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_label = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

        return predicted_label, confidence, probabilities[0].numpy()

    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, float, np.ndarray]]:
        """
        批量预测图像

        Args:
            image_paths: 图像路径列表

        Returns:
            预测结果列表，每个元素为 (predicted_label, confidence, probabilities)
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"预测图像 {image_path} 时出错: {str(e)}")
                results.append((-1, 0.0, np.zeros(10)))

        return results


def setup_callbacks() -> List[pl.Callback]:
    """
    设置训练回调函数

    Returns:
        回调函数列表
    """
    # 模型检查点 - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='mnist-transformer-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
        save_last=True,
        verbose=True
    )

    # 早停
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        mode='max',
        verbose=True
    )

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return [checkpoint_callback, early_stop_callback, lr_monitor]


def setup_logger() -> CSVLogger:
    """
    设置日志记录器

    Returns:
        CSV日志记录器
    """
    return CSVLogger(save_dir='logs', name='mnist_transformer')


def main() -> Tuple[ImageTransformerClassifier, MNISTImageDataModule]:
    """
    主训练函数

    Returns:
        model: 训练好的模型
        data_module: 数据模块
    """
    # 设置随机种子保证可复现性
    pl.seed_everything(42)

    print("=== MNIST手写数字识别训练开始 ===")

    # 第一步：设置数据模块
    print("\n=== 步骤1: 设置数据加载器 ===")
    data_module = MNISTImageDataModule(
        data_dir='/Users/cuijinlong/Documents/datasets/mnist',
        batch_size=128,
        num_workers=0,  # 在CPU上设置为0避免多进程问题
        validation_split=0.2
    )

    # 第二步：创建模型
    print("\n=== 步骤2: 创建Transformer模型 ===")
    model = ImageTransformerClassifier(
        num_classes=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout_rate=0.1
    )

    # 第三步：设置训练器
    print("\n=== 步骤3: 配置训练器 ===")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',
        devices=1,
        callbacks=setup_callbacks(),
        logger=setup_logger(),
        log_every_n_steps=20,
        enable_progress_bar=True,
        deterministic=True,  # 保证可复现性
        val_check_interval=0.5,  # 每个epoch验证两次
    )

    # 第四步：训练模型
    print("\n=== 步骤4: 开始训练 ===")
    trainer.fit(model, data_module)

    # 第五步：测试模型
    print("\n=== 步骤5: 测试模型 ===")
    test_results = trainer.test(model, data_module)
    print(f"测试准确率: {test_results[0]['test_acc']:.4f}")

    # 第六步：保存模型
    print("\n=== 步骤6: 保存模型 ===")
    model_path = 'transformer_image_classifier.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存为 '{model_path}'")

    # 第七步：演示预测
    print("\n=== 步骤7: 单张图像预测演示 ===")
    demo_prediction(model, data_module)

    print("\n=== 训练完成! ===")
    return model, data_module


def demo_prediction(model: ImageTransformerClassifier, data_module: MNISTImageDataModule):
    """
    演示预测功能

    Args:
        model: 训练好的模型
        data_module: 数据模块
    """
    test_dataset = data_module.test_dataset

    # 随机选择几个样本进行预测演示
    import random
    sample_indices = random.sample(range(len(test_dataset)), 3)

    model.eval()
    for i, idx in enumerate(sample_indices):
        sample_image, true_label = test_dataset[idx]

        with torch.no_grad():
            logits = model(sample_image.unsqueeze(0))
            probabilities = F.softmax(logits, dim=1)
            predicted_label = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

        print(f"\n样本 {i + 1}:")
        print(f"  真实标签: {true_label}")
        print(f"  预测标签: {predicted_label}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  是否正确: {'✓' if predicted_label == true_label else '✗'}")

        # 显示前3个最可能的类别
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        print("  最可能类别:")
        for j, (prob, cls_idx) in enumerate(zip(top_probs, top_indices)):
            print(f"    类别 {cls_idx}: {prob:.4f}")


def train_with_custom_config(
        data_dir: str = '/Users/cuijinlong/Documents/datasets/mnist',
        batch_size: int = 128,
        max_epochs: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        learning_rate: float = 1e-3
) -> Tuple[ImageTransformerClassifier, MNISTImageDataModule]:
    """
    使用自定义配置进行训练

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        learning_rate: 学习率

    Returns:
        model: 训练好的模型
        data_module: 数据模块
    """
    print(f"=== 使用自定义配置训练 ===")
    print(f"数据目录: {data_dir}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {max_epochs}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {nhead}")
    print(f"Transformer层数: {num_layers}")
    print(f"学习率: {learning_rate}")

    # 设置随机种子
    pl.seed_everything(42)

    # 数据模块
    data_module = MNISTImageDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    # 模型
    model = ImageTransformerClassifier(
        num_classes=10,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    # 训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=setup_callbacks(),
        logger=setup_logger(),
        enable_progress_bar=True
    )

    # 训练和测试
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    return model, data_module


if __name__ == "__main__":
    # 标准训练
    model, data_module = main()

    # 创建预测器实例
    # predictor = MNISTPredictor(model=model)
    # predictor = MNISTPredictor(model_path='transformer_image_classifier.pth')
    # label, confidence, probs = predictor.predict('/Users/cuijinlong/Documents/datasets/mnist/val/6.png')
    # print(f'预测结果: {label}, 置信度: {confidence:.4f}')

    print("\n" + "=" * 50)
    print("使用说明:")
    print("=" * 50)
    print("1. 预测单张图像:")
    print("""
    predictor = MNISTPredictor(model_path='transformer_image_classifier.pth')
    label, confidence, probs = predictor.predict('your_image.png')
    print(f'预测结果: {label}, 置信度: {confidence:.4f}')
    """)


    print("\n2. 批量预测:")
    print("""
    image_paths = ['img1.png', 'img2.png', 'img3.png']
    results = predictor.predict_batch(image_paths)
    for path, (label, conf, _) in zip(image_paths, results):
        print(f'{path}: 预测={label}, 置信度={conf:.4f}')
    """)

    print("\n3. 自定义配置训练:")
    print("""
    model, data_module = train_with_custom_config(
        data_dir='/path/to/your/data',
        batch_size=64,
        max_epochs=15,
        d_model=256,
        learning_rate=2e-3
    )
    """)