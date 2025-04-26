## 学习函数 y = x^3 + 2X^2 -x + 1

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 创建训练数据
# Uniform [-10,10]
x_train = torch.rand(1000, 1) * 20 - 10  
y_train = x_train**3 + 2 * x_train**2 - x_train + 1

print(x_train.shape, y_train.shape)


# 创建模型
# 至少加一个隐藏层，用 激活函数ReLU
# 在这个例子中，我们需要使用神经网络来学习一个非线性函数 y = x^3 + 2x^2 - x + 1
# 由于这是一个非线性函数，我们需要使用带有非线性激活函数的神经网络
# 我们将创建一个带有隐藏层的简单神经网络，并使用ReLU激活函数
# ReLU(x) = max(0, x)，它能够引入非线性，使网络能够学习复杂的函数关系

# 网络结构:
# 输入层(1个神经元) -> 隐藏层(10个神经元) -> ReLU激活函数 -> 输出层(1个神经元)
# 隐藏层的神经元数量(10)是一个超参数，可以根据问题复杂度调整
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleNet()

print(model)


# 定义损失函数
loss_fn = nn.MSELoss()  # 均方误差损失函数

# 定义优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器 学习率0.01   
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 训练过程
losses = []
for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# 生成测试数据
x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
y_test = x_test**3 + 2*x_test**2 - x_test + 1

# 预测
y_pred = model(x_test).detach()

# 绘制预测与真实对比
plt.plot(x_test.numpy(), y_test.numpy(), label='True Function')
plt.plot(x_test.numpy(), y_pred.numpy(), label='Model Prediction')
plt.legend()
plt.title('Model Fit')
plt.show()
