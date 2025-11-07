from convert_mnist_to_images_and_csv import convert_mnist_to_images_and_csv
from model import ImageTransformerClassifier
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from dataset import MNISTImageDataModule

def main():
    # 第一步：转换数据集
    print("=== 步骤1: 转换MNIST数据集为图片和CSV ===")
    # output_dir = convert_mnist_to_images_and_csv()
    
    # 第二步：设置数据模块和模型
    print("\n=== 步骤2: 设置数据加载器和模型 ===")
    data_module = MNISTImageDataModule(data_dir='/Users/cuijinlong/Documents/datasets/mnist', batch_size=128)
    
    model = ImageTransformerClassifier(
        num_classes=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        learning_rate=1e-3
    )
    
    # 第三步：训练模型
    print("\n=== 步骤3: 训练Transformer模型 ===")
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True
    )
    
    trainer.fit(model, data_module)
    
    # 第四步：测试模型
    print("\n=== 步骤4: 测试模型 ===")
    trainer.test(model, data_module)
    
    # 第五步：保存模型
    print("\n=== 步骤5: 保存模型 ===")
    torch.save(model.state_dict(), 'transformer_image_classifier.pth')
    print("模型已保存为 'transformer_image_classifier.pth'")
    
    # 第六步：演示单张图像预测
    print("\n=== 步骤6: 单张图像预测演示 ===")
    # 加载一张测试图像
    test_dataset = data_module.test_dataset
    sample_idx = 0  # 使用第一张测试图像
    sample_image, true_label = test_dataset[sample_idx]
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(sample_image.unsqueeze(0))  # 添加batch维度
        probabilities = F.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1).item()
    
    print(f"真实标签: {true_label}")
    print(f"预测标签: {predicted_label}")
    print(f"预测概率: {[f'{p:.4f}' for p in probabilities[0].numpy()]}")
    
    return model, data_module

# 预测单张外部图像的函数
def predict_external_image(model, image_path, transform=None):
    """
    预测单张外部图像
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # 加载图像
    image = Image.open(image_path)
    
    # 预处理
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
    
    return predicted_label, confidence, probabilities[0].numpy()

if __name__ == "__main__":
    model, data_module = main()
    
    # 演示如何使用预测函数
    print("\n=== 预测外部图像使用示例 ===")
    print("您可以使用以下函数预测自己的手写数字图像:")
    print("""
    predicted_label, confidence, all_probs = predict_external_image(
        model, 
        'your_image_path.png',
        transform=None
    )
    """)