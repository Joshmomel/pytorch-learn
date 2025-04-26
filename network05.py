# y = x1^2 + 2X2^3 - 3X1X2 + 4

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchinfo import summary



# 创建训练数据
x_train = torch.rand(1000, 1) * 2 - 1  # Uniform [-1,1]
x_train2 = torch.rand(1000, 1) * 2 - 1
y_train = x_train**2 + 2*x_train2**3 - 3*x_train*x_train2 + 4

# 将两个特征拼接为一个张量
X = torch.cat([x_train, x_train2], dim=1)  # 形状为 [1000, 2]

print(x_train.shape, x_train2.shape, y_train.shape, X.shape)


# 创建模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = SimpleNet()

print(model)


# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 训练过程
losses = []
for epoch in range(2000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())    


    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# 生成测试数据
x_test = torch.rand(1000, 1) * 2 - 1   # 形状为 [100, 1]
x_test2 = torch.rand(1000, 1) * 2 - 1  # 形状为 [100, 1]
X_test = torch.cat([x_test, x_test2], dim=1)  # 形状为 [100, 2] 

# 计算测试数据的真实值
y_test = x_test**2 + 2*x_test2**3 - 3*x_test*x_test2 + 4

# 预测
y_pred = model(X_test).detach()  # 形状为 [100, 1]


# 绘制预测与真实对比
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_test.numpy(), x_test2.numpy(), y_test.numpy(), label='True', color='blue')
ax.set_title('True Function')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(x_test.numpy(), x_test2.numpy(), y_pred.numpy(), label='Prediction', color='red')
ax.set_title('Model Prediction')

plt.show() 

summary(model, input_size=(1, 2))
