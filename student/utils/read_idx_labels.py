import numpy as np
import matplotlib.pyplot as plt

def read_idx_labels(filename):
    """读取 IDX 标签文件"""
    with open(filename, 'rb') as f:
        # 读取魔数
        magic_number = int.from_bytes(f.read(4), 'big')
        # 读取标签数量
        num_items = int.from_bytes(f.read(4), 'big')

        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels, num_items


# 读取标签数据
labels, num_labels = read_idx_labels('/tmp/data/MNIST/MNIST/raw/train-labels-idx1-ubyte')

print(f"标签数量: {num_labels}")
print(f"前50个标签: {labels[:50]}")
print(f"标签数据类型: {labels.dtype}")
print(f"标签取值范围: {np.unique(labels)}")

# 统计每个标签的数量
unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))
print("每个标签的数量:", label_counts)

# 可视化标签分布
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('数字')
plt.ylabel('数量')
plt.title('测试集标签分布')

plt.subplot(1, 2, 2)
plt.hist(labels, bins=10, rwidth=0.8)
plt.xlabel('数字')
plt.ylabel('频率')
plt.title('测试集标签直方图')
plt.tight_layout()
plt.show()