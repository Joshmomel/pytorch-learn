from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
california = fetch_california_housing()
X = california.data
y = california.target

# 标准化输入特征
scaler = StandardScaler()
# 使用StandardScaler对特征进行标准化处理
# 这一步将所有特征转换为均值为0、标准差为1的分布
# fit_transform方法先计算训练数据的均值和方差，再基于计算出的均值和方差进行数据转换
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
# 使用unsqueeze(1)是为了将一维张量转换为二维张量（列向量）
# 原始的y_train是一维数组，形状为[n_samples]
# 神经网络模型的输出是二维张量，形状为[n_samples, 1]
# 为了使标签y_train与模型输出形状匹配，需要将其转换为[n_samples, 1]的形状
# 这样在计算损失时，可以正确地进行元素级别的比较
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(X_train.shape, y_train.shape) 

# 创建模型
# 至少两层隐藏层，带ReLU
class CaliforniaHousingModel(nn.Module):
    def __init__(self):
        super(CaliforniaHousingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),  # 输入特征维度为8（加州房价数据集有8个特征）
            nn.ReLU(),         # 使用ReLU激活函数引入非线性
            nn.Linear(64, 32), # 第一个隐藏层64个神经元，第二个隐藏层32个神经元（逐层减半是常见设计模式）
            nn.ReLU(),         # 第二个隐藏层后的激活函数
            nn.Linear(32, 1)   # 输出层，输出维度为1（预测房价是单一数值）
        )
    
    def forward(self, x):
        return self.model(x)
    
model = CaliforniaHousingModel()

print(model)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器
# Adam优化器，初始学习率0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练过程
losses = []
train_time = 10000
last_learning_rate = 0.001
for epoch in range(train_time):
    # 在训练过程中途（一半左右）降低学习率以实现更精细的收敛
    if epoch == train_time/2:
        # 将学习率从0.01降低到0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = last_learning_rate
        print("Learning rate changed to 0.001")
    
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# 绘制预测与真实对比
y_pred = model(X_test).detach()

# 绘制预测与真实对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # 添加对角线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True)
plt.show()

# 用模型在测试集上预测
y_pred_test = model(X_test).detach().numpy()
y_true_test = y_test.numpy()

# 1. MSE
mse = mean_squared_error(y_true_test, y_pred_test)

# 2. RMSE
rmse = np.sqrt(mse)

# 3. MAE
mae = mean_absolute_error(y_true_test, y_pred_test)

# 4. R2 Score
r2 = r2_score(y_true_test, y_pred_test)

# 打印结果
print(f"Test MSE : {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test R²  : {r2:.4f}")