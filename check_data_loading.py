import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

# --- 这里粘贴之前定义的 CIFAR100C 类 ---
class CIFAR100C(torch.utils.data.Dataset):
    def __init__(self, root_dir, corruption_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        file_path = os.path.join(root_dir, f'{corruption_type}.npy')
        labels_path = os.path.join(root_dir, 'labels.npy')
        
        if not os.path.exists(file_path): raise RuntimeError(f"没找到: {file_path}")
        if not os.path.exists(labels_path): raise RuntimeError(f"没找到: {labels_path}")
            
        self.data = np.load(file_path)
        self.targets = np.tile(np.load(labels_path), 5).astype(np.int64)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        return img, target

# --- 检查脚本 ---
def check():
    print(">>> 开始数据自检...")
    
    # 1. 检查路径 (根据你的截图调整)
    clean_root = './data/cifar100'
    corrupt_root = './data/cifar100-c'
    
    # 2. 检查 Clean Data
    try:
        clean_ds = datasets.CIFAR100(root=clean_root, train=False, download=False)
        print(f"✅ Clean CIFAR-100 读取成功! 数量: {len(clean_ds)}")
    except Exception as e:
        print(f"❌ Clean CIFAR-100 读取失败: {e}")
        return

    # 3. 检查 Corrupt Data
    try:
        # 测试读取 'fog' 类型
        corrupt_ds = CIFAR100C(root_dir=corrupt_root, corruption_type='fog')
        print(f"✅ CIFAR-100-C (Fog) 读取成功! 数量: {len(corrupt_ds)}")
        
        # 验证形状
        img, label = corrupt_ds[0]
        print(f"   样本形状: {img.size} (应为 32x32)")
        print(f"   样本标签: {label}")
        
    except Exception as e:
        print(f"❌ CIFAR-100-C 读取失败: {e}")
        print(f"   请检查路径: {corrupt_root}")

if __name__ == '__main__':
    check()