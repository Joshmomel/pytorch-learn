## 学习函数 y = 2X1 + 3X2 + 5

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 创建训练数据
# 生成100个样本，每个样本有2个特征
# 特征1在[-1, 1]之间均匀分布
# 特征2在[-1, 1]之间均匀分布
# 标签是特征1和特征2的线性组合加上一个常数

x_train = torch.rand(1000, 1) * 2 - 1  # Uniform [-1,1]
x_train2 = torch.rand(1000, 1) * 2 - 1
# 将两个特征拼接为一个张量
x_combined = torch.cat((x_train, x_train2), dim=1)
y_train = 2 * x_train + 3 * x_train2 + 5

print(x_train.shape, x_train2.shape, y_train.shape)
print("x_combined shape:", x_combined.shape)


# 创建模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet()

print(model)


# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
for epoch in range(1000):
    y_pred = model(x_combined)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 查看学到的权重和偏置
print(f"Learned weights: {model.linear.weight.detach().numpy()}, bias: {model.linear.bias.item():.4f}")

# 绘制Loss曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

