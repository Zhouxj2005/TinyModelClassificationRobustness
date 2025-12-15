# ================= 修复补丁开始 =================
import skimage.filters
# 保存原始的 gaussian 函数
_old_gaussian = skimage.filters.gaussian

def _fixed_gaussian(*args, **kwargs):
    # 如果检测到旧参数 'multichannel'
    if 'multichannel' in kwargs:
        # 取出它的值，并从 kwargs 中删除
        is_multichannel = kwargs.pop('multichannel')
        # 如果是 True，这就对应新版的 channel_axis=-1 (表示最后一个维度是通道)
        if is_multichannel:
            kwargs['channel_axis'] = -1
        else:
            kwargs['channel_axis'] = None
    # 调用原始函数，但传入修改后的参数
    return _old_gaussian(*args, **kwargs)

# 将 skimage 里的函数替换成我们的“修正版”
skimage.filters.gaussian = _fixed_gaussian
# ================= 修复补丁结束 =================

import numpy as np
import torch
import torchvision
import os
from imagecorruptions import corrupt
# 忽略那个烦人的 pkg_resources 警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def generate_cifar100_c(source_dir='./data/cifar100', target_dir='./data/cifar100-c'):
    print("正在生成 CIFAR-100-C 数据集，这可能需要几分钟...")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 1. 加载干净的 CIFAR-100 测试集
    print("加载原始 CIFAR-100 测试集...")
    testset = torchvision.datasets.CIFAR100(root=source_dir, train=False, download=True)
    data = testset.data # Shape: (10000, 32, 32, 3) - uint8
    labels = testset.targets

    # 保存标签
    np.save(os.path.join(target_dir, 'labels.npy'), np.array(labels))
    print(f"标签已保存至 {target_dir}/labels.npy")

    # 2. 定义 15 种测试腐蚀类型
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    # 3. 开始生成
    total_corruptions = len(corruption_types)
    for i, c_name in enumerate(corruption_types):
        print(f"[{i+1}/{total_corruptions}] 正在生成: {c_name} ...")
        
        all_severity_images = []
        
        # 对每个严重等级 (1-5) 进行循环
        for severity in range(1, 6):
            # ===【关键修改点】===
            # 不能直接传整个 data，必须一张一张处理
            corrupted_batch = []
            for img in data:
                # img: (32, 32, 3) -> 处理 -> (32, 32, 3)
                out = corrupt(img, severity=severity, corruption_name=c_name)
                corrupted_batch.append(out)
            
            # 转换回 numpy 数组
            corrupted_batch = np.array(corrupted_batch, dtype=np.uint8)
            all_severity_images.append(corrupted_batch)
            # ===================
        
        # 拼接数据: 变成 (50000, 32, 32, 3)
        final_array = np.vstack(all_severity_images)
        
        # 保存为 .npy 文件
        save_path = os.path.join(target_dir, f'{c_name}.npy')
        np.save(save_path, final_array)
        
    print("✅ CIFAR-100-C 生成完成！")

if __name__ == "__main__":
    generate_cifar100_c()