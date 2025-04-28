import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# The transform below will be used to:
# 1. Convert images to PyTorch tensors
# 2. Normalize pixel values from [0,1] to [-1,1] range
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层
        # 定义卷积层部分
        # 第一个卷积层:
        # - 输入通道为1（灰度图像）
        # - 输出32个特征图
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            #归一化/正则化(稳定&防过拟合)
            nn.BatchNorm2d(32),
            #激活函数:
            nn.ReLU(),  
            #池化/步幅(降采样):
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积层:
            # - 输入通道为32（来自第一个卷积层的输出）
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        # 全连接层
        # 定义全连接层部分
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

model = SimpleCNN()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # model.train() 设置模型为训练模式
    # 这会启用批归一化和dropout等层的训练行为
    # 在训练模式下:
    # 1. BatchNorm层会更新其运行均值和方差统计信息
    # 2. Dropout层会随机丢弃神经元
    # 3. 梯度会被计算和更新
    # 相比之下，model.eval()会设置为评估模式，禁用这些训练特定的行为
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = loss_fn(outputs, labels)
        # 在反向传播前清零梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 测试评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    test_accuracies.append(acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Test Acc: {acc*100:.2f}%")