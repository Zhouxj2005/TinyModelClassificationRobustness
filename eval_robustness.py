import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from models import get_model
from check_data_loading import CIFAR100C # 复用之前写的 Dataset 类

# --- 配置参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet_v2', type=str)
parser.add_argument('--checkpoint', default='./checkpoints/mobilenet_v2_standard.pth', type=str)
args = parser.parse_args()

def eval_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始评估鲁棒性: {args.model} ===")
    
    # 1. 加载模型
    net = get_model(args.model, num_classes=100)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: 找不到权重文件 {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()
    
    print(f"Loaded Checkpoint: Clean Acc = {checkpoint['acc']:.2f}%")

    # 2. 定义腐蚀类型 (Hendrycks Standard 15 types)
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    # 3. 评估循环
    # 依然需要标准化，因为模型是用标准化数据训练的
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    corruption_accs = []

    with torch.no_grad():
        for c_name in corruption_types:
            # 加载特定腐蚀的数据集
            dataset = CIFAR100C(root_dir='./data/cifar100-c', corruption_type=c_name, transform=transform)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
            
            correct = 0
            total = 0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            corruption_accs.append(acc)
            print(f"Corruption: {c_name:20s} | Acc: {acc:.2f}%")

    # 4. 汇总结果
    mean_robust_acc = np.mean(corruption_accs)
    print("-" * 40)
    print(f"Clean Accuracy:  {checkpoint['acc']:.2f}%")
    print(f"Robust Accuracy: {mean_robust_acc:.2f}% (Average of 15 corruptions)")
    print("-" * 40)

if __name__ == "__main__":
    eval_robustness()