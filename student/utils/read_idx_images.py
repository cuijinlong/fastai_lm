import numpy as np
import matplotlib.pyplot as plt


def read_idx(filename):
    """读取 IDX 文件"""
    with open(filename, 'rb') as f:
        # 读取魔数
        magic_number = int.from_bytes(f.read(4), 'big')
        # 读取维度数量
        num_dimensions = magic_number & 0xff
        # 读取各维度大小
        dimensions = []
        for i in range(num_dimensions):
            dimensions.append(int.from_bytes(f.read(4), 'big'))

        # 读取数据
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(dimensions)

    return data


# 读取图像数据
images = read_idx('/tmp/data/MNIST/MNIST/raw/train-images-idx3-ubyte')

print(f"图像数量: {images.shape[0]}")
print(f"图像尺寸: {images.shape[1]} x {images.shape[2]}")

# 显示前几张图像
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'图像 {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# 查看单个图像的像素值
print("第一张图像的像素值:")
print(images[0])