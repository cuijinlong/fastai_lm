import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import math

def convert_mnist_to_images_and_csv(data_dir='./mnist_data', output_dir='./mnist_images_csv'):
    """
    将MNIST数据集转换为图片和CSV文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # 定义转换（只转换为Tensor，不标准化，因为我们要保存原始图片）
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 加载MNIST数据集
    print("下载并加载MNIST数据集...")
    train_dataset = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform)
    
    # 准备存储数据的列表
    train_data = []
    test_data = []
    
    # 处理训练集
    print("处理训练集...")
    for i, (image_tensor, label) in enumerate(train_dataset):
        # 转换为PIL图像
        image_np = image_tensor.squeeze().numpy() * 255  # 反归一化到0-255
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        
        # 保存图像
        filename = f"train_{i:05d}.png"
        filepath = os.path.join(output_dir, 'train', filename)
        image_pil.save(filepath)
        
        # 记录到数据列表
        train_data.append({
            'filename': filename,
            'label': label,
            'split': 'train'
        })
        
        if i % 1000 == 0:
            print(f"已处理 {i}/60000 训练图像")
    
    # 处理测试集
    print("处理测试集...")
    for i, (image_tensor, label) in enumerate(test_dataset):
        # 转换为PIL图像
        image_np = image_tensor.squeeze().numpy() * 255
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        
        # 保存图像
        filename = f"test_{i:05d}.png"
        filepath = os.path.join(output_dir, 'test', filename)
        image_pil.save(filepath)
        
        # 记录到数据列表
        test_data.append({
            'filename': filename,
            'label': label,
            'split': 'test'
        })
        
        if i % 1000 == 0:
            print(f"已处理 {i}/10000 测试图像")
    
    # 创建DataFrame并保存为CSV
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # 保存CSV文件
    train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    
    # 创建合并的CSV文件
    all_data = train_data + test_data
    all_df = pd.DataFrame(all_data)
    all_df.to_csv(os.path.join(output_dir, 'all_labels.csv'), index=False)
    
    print(f"转换完成！")
    print(f"训练集: {len(train_data)} 张图片")
    print(f"测试集: {len(test_data)} 张图片")
    print(f"图片保存在: {output_dir}")
    print(f"标签文件: {output_dir}/*.csv")
    
    return output_dir

# 运行转换
if __name__ == "__main__":
    output_dir = convert_mnist_to_images_and_csv()