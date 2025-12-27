import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from models import get_model
from check_data_loading import CIFAR100C 

# --- 配置参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet18_cifar', type=str)
parser.add_argument('--checkpoint', default='./final_checkpoint/resnet18-sota.pth', type=str)
args = parser.parse_args()

def eval_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始评估鲁棒性: {args.model} ===")
    
    # 1. 加载模型结构
    # resnet18_cifar 返回的是 Raw Model
    # mobilenet_v2 返回的是 NormalizedModel (Wrapper)
    net = get_model(args.model, num_classes=100)
    net = net.to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: 找不到权重文件 {args.checkpoint}")
        return
        
    print(f"=> Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # --- 智能提取 state_dict ---
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
            acc = checkpoint.get('acc', 0.0)
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            acc = 0.0
        else:
            state_dict = checkpoint
            acc = 0.0
    else:
        state_dict = checkpoint
        acc = 0.0

    # 🚑 修复 acc 为 None 的情况
    if acc is None:
        acc = 0.0
    print(f"   Recorded Acc in Checkpoint: {acc:.2f}%")

    # --- 🧠 核心修复：自动适配 Teacher/Student 和 新/旧权重 ---
    
    # 判断当前网络是不是包装器 (Student 是 NormalizedModel，Teacher 是 ResNetRobustBench)
    is_wrapper_model = hasattr(net, 'model') 
    
    # 判断权重文件是否带包装前缀 (model.xxx)
    ckpt_keys = list(state_dict.keys())
    ckpt_has_wrapper_prefix = any(k.startswith('model.') for k in ckpt_keys)
    
    msg = ""
    try:
        if is_wrapper_model:
            # === 情况 A: 模型是 Student (NormalizedModel) ===
            if ckpt_has_wrapper_prefix:
                # 权重也是带包装的 -> 直接加载
                msg = net.load_state_dict(state_dict, strict=False)
                print("   [Load] Student Model (Wrapped) <- Wrapped Checkpoint")
            else:
                # 权重是裸的 -> 加载到内部 net.model
                # 去掉可能的 module. 前缀
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                msg = net.model.load_state_dict(new_state_dict, strict=True) # 内部必须严格匹配
                print("   [Load] Student Model (Inner) <- Raw Checkpoint")
        else:
            # === 情况 B: 模型是 Teacher (ResNetRobustBench) ===
            if ckpt_has_wrapper_prefix:
                # 权重带包装 -> 去掉 'model.' 前缀再加载
                new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                msg = net.load_state_dict(new_state_dict, strict=False)
                print("   [Load] Teacher Model (Raw) <- Wrapped Checkpoint (Stripped)")
            else:
                # 权重也是裸的 -> 直接加载
                # 去掉 module. 前缀以防万一
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                msg = net.load_state_dict(new_state_dict, strict=False)
                print("   [Load] Teacher Model (Raw) <- Raw Checkpoint")
                
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        print("尝试打印前5个Key帮助调试:")
        print(ckpt_keys[:5])
        return

    print(f"   Load Msg: {msg}")
    net.eval()

    # 2. 定义腐蚀类型
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    # 3. 评估循环
    # 不管是 Teacher 还是 Student，模型内部都处理了归一化 (models.py里定义的)
    # 所以这里只需要 ToTensor
    transform = transforms.Compose([transforms.ToTensor()])

    corruption_accs = []

    with torch.no_grad():
        for c_name in corruption_types:
            dataset = CIFAR100C(root_dir='./data/cifar100-c', corruption_type=c_name, transform=transform)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)
            
            correct = 0; total = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc_c = 100. * correct / total
            corruption_accs.append(acc_c)
            print(f"Corruption: {c_name:20s} | Acc: {acc_c:.2f}%")

    # 4. 汇总
    mean_robust_acc = np.mean(corruption_accs)
    print("-" * 40)
    print(f"Checkpoint Clean Acc: {acc:.2f}%")
    print(f"Robust Accuracy (mAcc): {mean_robust_acc:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    eval_robustness()