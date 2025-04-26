import torch
import torch.nn as nn
import torch.optim as optim


# 输入数据 x 和对应标签 y
# 创建训练数据：
# 1. torch.linspace(-1, 1, 100) 生成一个包含100个元素的一维张量，这些元素均匀分布在-1到1之间
# 2. torch.unsqueeze(tensor, dim=1) 在第1维（列维度）上增加一个维度
# 3. 最终得到形状为[100,1]的二维张量，即100行1列，每行是一个样本特征
x_train = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # shape [100,1]
y_train = 2 * x_train + 3 

print(x_train.shape, y_train.shape)


# 创建模型
class SimpleNet(nn.Module):
    """
    一个简单的线性回归神经网络模型
    """
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        super(SimpleNet, self).__init__()
        # 创建一个线性层，输入特征维度为1，输出维度为1
        # 这适合我们的数据，因为x_train的形状是[100,1]，每个样本只有一个特征
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 前向传播函数，将输入x通过线性层转换
        # 对于线性回归，这就是 y = wx + b 的计算过程
        return self.linear(x)


# 创建模型实例
model = SimpleNet()

# 打印模型结构
print(model)



# 定义损失函数
# nn.MSELoss() 创建均方误差损失函数实例
# 均方误差(Mean Squared Error)是回归问题中常用的损失函数
# 它计算预测值与真实值之间差的平方的平均值
# 公式: MSE = (1/n) * Σ(y_pred - y_true)²
# 其中n是样本数量，y_pred是模型预测值，y_true是真实标签值

loss_fn = nn.MSELoss()

# 定义优化器
# optim.SGD 创建随机梯度下降(Stochastic Gradient Descent)优化器
# model.parameters() 获取模型中所有需要优化的参数
# lr=0.01 设置学习率为0.01，控制每次参数更新的步长
# 优化器的作用是根据计算的梯度更新模型参数，使损失函数值不断减小
optimizer = optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1000):
    # 前向传播
    # 将输入数据x_train传递给模型，得到预测值y_pred
    # 模型会根据输入数据和当前参数计算输出
    # 这里y_pred的形状与y_train相同，都是[100,1]
    y_pred = model(x_train)
    
    # 计算损失
    loss = loss_fn(y_pred, y_train)
    
  
    # 在每次参数更新前清零梯度
    # 这是必要的，因为PyTorch会默认累积梯度
    # 如果不清零，当前计算的梯度会被加到已存在的梯度上
    # 这会导致梯度累积，使优化过程不正确
    # 特别是在批量训练时，每个批次都需要独立的梯度计算
    optimizer.zero_grad()

    # 反向传播
    # loss.backward() 计算损失函数关于模型参数的梯度
    # 这一步使用链式法则计算每个参数对损失函数的影响程度
    # 梯度表示损失函数在当前参数值处的斜率
    # 正梯度表示增加参数值会增加损失，负梯度表示增加参数值会减少损失
    # PyTorch的自动微分系统会自动处理这个复杂的计算过程
    loss.backward()

    # 更新参数
    # optimizer.step() 使用优化器更新模型参数
    # 这一步根据计算的梯度更新模型参数
    # 梯度下降法通过沿着梯度的反方向更新参数来最小化损失函数
    # 优化器会自动调整参数，使损失函数不断减小
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 查看学到的权重和偏置
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"Learned weight: {w:.4f}, bias: {b:.4f}")
