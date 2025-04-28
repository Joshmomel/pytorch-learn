import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义与原模型相同的变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def visualize_mnist_samples(num_samples=10, rows=2):
    """显示MNIST数据集中的样本图像"""
    cols = num_samples // rows
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        img, label = train_dataset[i]
        # 反归一化图像
        img = img / 2 + 0.5  
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(f'Digit: {label}')
        plt.axis('off')
    
    plt.suptitle('MNIST Handwritten Digit Samples', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_class_examples():
    """显示每个类别(0-9)的样本图像"""
    plt.figure(figsize=(15, 8))
    
    # 为每个数字0-9找到一个示例
    for digit in range(10):
        # 查找带有此标签的第一个样本
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            if label == digit:
                img, _ = train_dataset[i]
                # 反归一化图像
                img = img / 2 + 0.5
                
                plt.subplot(2, 5, digit+1)
                plt.imshow(img.squeeze().numpy(), cmap='gray')
                plt.title(f'Digit: {digit}')
                plt.axis('off')
                break
    
    plt.suptitle('Sample of Each Digit from MNIST Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_batch(batch_size=64):
    """显示一个批次的图像"""
    # 创建数据加载器以获取批次
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # 获取一个批次
    images, labels = next(iter(data_loader))
    
    # 创建网格显示
    grid_size = int(batch_size ** 0.5)
    if grid_size ** 2 < batch_size:
        grid_size += 1
        
    plt.figure(figsize=(15, 15))
    for i in range(min(batch_size, 36)):  # 最多显示36个样本以避免图像过于拥挤
        img = images[i] / 2 + 0.5  # 反归一化
        plt.subplot(6, 6, i+1)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(f'{labels[i].item()}')
        plt.axis('off')
    
    plt.suptitle('MNIST Samples in a Batch', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading and visualizing MNIST dataset...")
    
    # 显示10个随机样本
    visualize_mnist_samples(10)
    
    # 显示每个数字类别的样本
    visualize_class_examples()
    
    # 显示一个批次的样本
    visualize_batch(36) 