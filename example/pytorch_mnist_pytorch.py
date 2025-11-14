import torch
from torch import nn
import torch.optim as optim
from mnist_net import MNISTNet
from dataset import MNISTImageDataModule

device = torch.device('cpu') # model、inputs、labels都放在 CPU 或 GPU 上去
if __name__ == '__main__':
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    data_module = MNISTImageDataModule(
        data_dir='/Users/cuijinlong/Documents/datasets/mnist_custom',
        batch_size=5,
        num_workers=0
    )
    data_module.setup()  # 初始化数据集

    # 获取训练数据加载器
    train_loader = data_module.train_dataloader()

    for epoch in range(10):
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad() # 清空梯度
            outputs =  model(inputs.to(device)) # 前向传插
            loss = criterion(outputs , labels.to(device)) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新权重
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item()}')