import torch

x = torch.rand(2, 3)
print(x)
print(x.dtype)
print(x.shape)
print(x.device)


# unsqueeze 方法用于在指定位置添加一个维度（大小为1的维度）
# 参数0表示在第0维（最外层）添加一个维度
# 例如，如果x的形状是[2, 3]，那么y = x.unsqueeze(0)后，y的形状变为[1, 2, 3]
y = x.unsqueeze(0)  # 在最外层添加一个维度
print(y)
print(y.shape)

y2 = y.unsqueeze(0)
print(y2)
print(y2.shape)




