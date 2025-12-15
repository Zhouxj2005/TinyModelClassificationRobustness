import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_clean_cifar100(root_dir='./data/cifar100'):
    print(f"正在准备 CIFAR-100 数据集，存放路径: {root_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # 训练集
    trainset = torchvision.datasets.CIFAR100(
        root=root_dir, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor() # 仅用于下载测试，实际训练会有更复杂的transform
    )
    
    # 测试集
    testset = torchvision.datasets.CIFAR100(
        root=root_dir, 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    print("✅ CIFAR-100 (Clean) 下载/校验完成！")

if __name__ == "__main__":
    download_clean_cifar100()