import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from models import get_model
import matplotlib.patches as patches

# 设置字体
plt.rcParams['font.family'] = 'serif'

# CIFAR-100-C 数据集加载器
class CIFAR100C(torch.utils.data.Dataset):
    def __init__(self, root_dir, corruption_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        file_path = os.path.join(root_dir, f'{corruption_type}.npy')
        labels_path = os.path.join(root_dir, 'labels.npy')
        
        if not os.path.exists(file_path):
            raise RuntimeError(f"未找到: {file_path}")
        if not os.path.exists(labels_path):
            raise RuntimeError(f"未找到: {labels_path}")
            
        self.data = np.load(file_path)
        self.targets = np.tile(np.load(labels_path), 5).astype(np.int64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def load_model_checkpoint(model_arch, checkpoint_path, device):
    """加载模型和权重"""
    net = get_model(model_arch, num_classes=100)
    net = net.to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 智能提取 state_dict
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 自动适配 wrapper 和裸模型
    is_wrapper = hasattr(net, 'model')
    has_wrapper_prefix = any(k.startswith('model.') for k in state_dict.keys())
    
    try:
        if is_wrapper:
            if has_wrapper_prefix:
                net.load_state_dict(state_dict, strict=False)
            else:
                net.model.load_state_dict(state_dict, strict=False)
        else:
            if has_wrapper_prefix:
                new_state = {k.replace('model.', ''): v for k, v in state_dict.items()}
                net.load_state_dict(new_state, strict=False)
            else:
                net.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"警告: 加载权重时出错: {e}")
    
    net.eval()
    return net


def get_predictions(model, images, device):
    """获取模型在一批图像上的预测"""
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


def evaluate_on_corruption(model, corruption_types, device, batch_size=128):
    """在各种腐蚀上评估模型的准确率"""
    transform = transforms.Compose([transforms.ToTensor()])
    results = {}
    
    for corruption in corruption_types:
        try:
            dataset = CIFAR100C('./data/cifar100-c', corruption, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            correct = 0
            total = 0
            with torch.no_grad():
                for images, targets in loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
            
            accuracy = 100.0 * correct / total
            results[corruption] = accuracy
            print(f"  {corruption:20s} -> Accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"  {corruption:20s} -> Error: {e}")
            results[corruption] = 0.0
    
    return results


def find_contrasting_samples(model_augmix, model_ours, device, corruption_types, max_samples=2):
    """
    寻找样本使得: Ours 正确，AugMix 错误
    """
    contrasting = []
    transform = transforms.Compose([transforms.ToTensor()])
    clean_ds = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, 
                                              transform=transforms.Compose([transforms.ToTensor()]))
    
    for corruption in corruption_types:
        if len(contrasting) >= max_samples:
            break
            
        print(f"  搜索 {corruption}...", end=' ')
        try:
            corrupt_ds = CIFAR100C('./data/cifar100-c', corruption, transform=transform)
            
            for idx in range(len(corrupt_ds)):
                if len(contrasting) >= max_samples:
                    break
                
                # 获取数据
                clean_img, clean_label = clean_ds[idx]
                corrupt_img, _ = corrupt_ds[idx]
                
                # 获取预测
                augmix_pred = get_predictions(model_augmix, corrupt_img.unsqueeze(0), device)[0]
                ours_pred = get_predictions(model_ours, corrupt_img.unsqueeze(0), device)[0]
                
                # 找到: Ours 正确，AugMix 错误
                if ours_pred == clean_label and augmix_pred != clean_label:
                    contrasting.append({
                        'idx': idx,
                        'corruption': corruption,
                        'clean_img': clean_img,
                        'corrupt_img': corrupt_img,
                        'label': clean_label,
                        'augmix_pred': augmix_pred,
                        'ours_pred': ours_pred
                    })
                    print(f"找到样本 {idx}")
                    break
        except Exception as e:
            print(f"错误: {e}")
    
    return contrasting


def draw_qualitative_figure():
    # ==========================================
    # 1. 加载模型和数据
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载 AugMix 模型
    print("\n=> 加载 AugMix 模型...")
    augmix_checkpoint = './checkpoint/mobilenet_v2_augmix.pth'
    model_augmix = load_model_checkpoint('mobilenet_v2', augmix_checkpoint, device)
    
    # 加载 Ours 模型（distill 改进版本）
    print("=> 加载 Ours (Distill Improved) 模型...")
    ours_checkpoint = './checkpoint/mobilenet_v2_distill_improved_best.pth'
    model_ours = load_model_checkpoint('mobilenet_v2', ours_checkpoint, device)
    
    # 选择评估的腐蚀类型（可视化用）
    corruption_types = ['gaussian_noise', 'defocus_blur', 'glass_blur', 'fog', 
                       'brightness', 'contrast', 'elastic_transform', 'pixelate',
                       'motion_blur', 'snow', 'frost', 'jpeg_compression']
    
    print("\n=> 评估 AugMix 模型...")
    results_augmix = evaluate_on_corruption(model_augmix, corruption_types, device)
    
    print("\n=> 评估 Ours 模型...")
    results_ours = evaluate_on_corruption(model_ours, corruption_types, device)
    
    print("\n=> 寻找对比样本 (Ours 正确, AugMix 错误)...")
    contrasting_samples = find_contrasting_samples(model_augmix, model_ours, device, corruption_types, max_samples=2)
    
    if len(contrasting_samples) == 0:
        print("❌ 未找到满足条件的样本！")
        print("正在生成准确率对比图表...")
        # 继续绘制对比图表
    
    # ==========================================
    # 2. 展示对比样本（如果找到的话）
    # ==========================================
    if len(contrasting_samples) > 0:
        fig, axes = plt.subplots(len(contrasting_samples), 3, figsize=(12, 5*len(contrasting_samples)), 
                                  gridspec_kw={'width_ratios': [1, 1, 1.2]})
        
        if len(contrasting_samples) == 1:
            axes = [axes]
        
        for row, sample in enumerate(contrasting_samples):
            clean_img = sample['clean_img']
            corrupt_img = sample['corrupt_img']
            corruption = sample['corruption']
            
            # 显示原始图像
            ax = axes[row][0]
            ax.imshow(clean_img.permute(1, 2, 0).numpy())
            ax.set_title("Original (Clean)", fontsize=10, color='gray')
            ax.axis('off')
            
            # 显示腐蚀图像
            ax = axes[row][1]
            ax.imshow(corrupt_img.permute(1, 2, 0).numpy())
            ax.set_title(f"Corrupted ({corruption})", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # 显示预测对比
            ax = axes[row][2]
            ax.axis('off')
            
            # AugMix 错误，Ours 正确
            augmix_correct = False  # 我们找的就是 AugMix 错误的
            ours_correct = True     # 且 Ours 正确
            
            plot_predictions(ax, augmix_correct, ours_correct)

        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/qualitative_results.pdf', bbox_inches='tight', dpi=150)
        print("Figure saved to outputs/qualitative_results.pdf")
    else:
        print("⚠️  未生成定性对比图表（未找到对比样本）")
    
    # ==========================================
    # 3. 绘制准确率对比图
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    corruptions = list(results_augmix.keys())
    augmix_accs = [results_augmix[c] for c in corruptions]
    ours_accs = [results_ours[c] for c in corruptions]
    
    x = np.arange(len(corruptions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, augmix_accs, width, label='AugMix', color='#2980b9', alpha=0.8)
    bars2 = ax.bar(x + width/2, ours_accs, width, label='Ours', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Corruption Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Comparison: AugMix vs Ours', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png', bbox_inches='tight', dpi=150)
    print("Accuracy comparison chart saved to outputs/accuracy_comparison.png")
    
    # 计算并打印平均准确率
    avg_augmix = np.mean(augmix_accs)
    avg_ours = np.mean(ours_accs)
    improvement = avg_ours - avg_augmix
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  AugMix Average Accuracy: {avg_augmix:.2f}%")
    print(f"  Ours Average Accuracy:   {avg_ours:.2f}%")
    print(f"  Improvement:             {improvement:+.2f}%")
    print("="*50)


def plot_predictions(ax, augmix_correct, ours_correct):
    """
    在指定的 ax 上画出红绿对比的预测结果
    """
    # 背景框
    rect = patches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05", 
                                  fc='#f9f9f9', ec='lightgray')
    ax.add_patch(rect)
    
    # AugMix Result
    augmix_status = "Correct" if augmix_correct else "Wrong"
    augmix_color = '#2ca02c' if augmix_correct else '#d62728'
    ax.text(0.1, 0.65, "AugMix:", fontsize=11, color='gray', fontweight='bold')
    ax.text(0.9, 0.65, augmix_status, ha='right', fontsize=11, fontweight='bold', color=augmix_color)
    
    ax.plot([0.1, 0.9], [0.58, 0.58], '-', color='gray', linewidth=0.5)
    
    # Ours Result
    ours_status = "Correct" if ours_correct else "Wrong"
    ours_color = '#2ca02c' if ours_correct else '#d62728'
    ax.text(0.1, 0.35, "Ours:", fontsize=11, color='gray', fontweight='bold')
    ax.text(0.9, 0.35, ours_status, ha='right', fontsize=11, fontweight='bold', color=ours_color)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


if __name__ == "__main__":
    draw_qualitative_figure()