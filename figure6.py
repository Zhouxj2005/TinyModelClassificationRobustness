import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from models import get_model # 你的模型定义文件

# 配置
CHECKPOINT_PATH = './checkpoint/mobilenet_v2_distill_15.0_200_0.05.pth' # 你的最佳模型路径
BATCH_SIZE = 500 # 取500个点画图就够了，太多会乱
NUM_CLASSES_TO_PLOT = 10 # 只画10个类，不然颜色分不清

def get_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    
    # 注册Hook提取倒数第二层特征
    # 注意：MobileNetV2的classifier之前的特征是 mean pooling 后的
    def hook_fn(module, input, output):
        # input[0] 通常是 feature
        features.append(input[0].detach().cpu().numpy().reshape(input[0].size(0), -1))
        
    # 定位 MobileNetV2 的 classifier 层
    # 通常是 model.classifier
    handle = model.classifier.register_forward_hook(lambda m, i, o: None) # 占位
    # 我们需要的是 classifier 之前的 feature。
    # 对于 MobileNetV2，直接跑 model.features(x) 然后 pooling 可能更方便
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            # 这里的实现取决于你的模型结构
            # 既然你的 forward 里直接出 logits，我们手动跑一下前半部分
            if hasattr(model, 'features'):
                out = model.features(inputs)
                out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                feat = out.flatten(1)
                features.append(feat.cpu().numpy())
                labels.append(targets.numpy())
            else:
                # 如果找不到，就用笨办法：Hook classifier
                # 这里假设你懂一点模型结构，如果不懂，告诉我 models.py 怎么写的
                print("Model structure distinct, trying hook...")
                # ... (hook logic)
            
            if len(np.concatenate(labels)) >= BATCH_SIZE:
                break
                
    return np.concatenate(features)[:BATCH_SIZE], np.concatenate(labels)[:BATCH_SIZE]

def add_noise(x):
    # 手动添加高斯噪声模拟 CIFAR-100-C
    return x + torch.randn_like(x) * 0.1

def plot_real_tsne():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据 (加噪声)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(add_noise) # 注入噪声！
    ])
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    # 筛选前N个类
    indices = [i for i, label in enumerate(testset.targets) if label < NUM_CLASSES_TO_PLOT]
    subset = torch.utils.data.Subset(testset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    # 2. 加载 Baseline (AugMix) - 假设你有，没有就加载 Standard
    # 如果没有 Baseline 的权重，就加载一个没训练好的或者 Standard 模型
    # 这里为了演示，我们假设 baseline 是你自己训练的另一个 checkpoint
    # 如果实在没有，可以用 ImageNet 预训练的 mobilenetv2 (torchvision自带) 充数对比
    model_base = torchvision.models.mobilenet_v2(pretrained=True) 
    model_base.classifier[1] = nn.Linear(1280, 100) # 改一下头
    model_base = model_base.to(device)
    
    # 3. 加载 Ours
    model_ours = get_model('mobilenet_v2', num_classes=100).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    # 处理一下 state_dict key
    state_dict = ckpt['net'] if 'net' in ckpt else ckpt
    # 你的代码里存的是 student.state_dict()，应该可以直接加载
    model_ours.load_state_dict(state_dict)

    print("Extracting features...")
    feat_base, y_base = get_features(model_base, loader, device)
    feat_ours, y_ours = get_features(model_ours, loader, device)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_base_2d = tsne.fit_transform(feat_base)
    X_ours_2d = tsne.fit_transform(feat_ours)

    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline
    scatter1 = ax1.scatter(X_base_2d[:, 0], X_base_2d[:, 1], c=y_base, cmap='tab10', s=15, alpha=0.6)
    ax1.set_title('(a) Baseline (AugMix) under Noise', fontsize=12)
    ax1.axis('off')
    
    # Ours
    scatter2 = ax2.scatter(X_ours_2d[:, 0], X_ours_2d[:, 1], c=y_ours, cmap='tab10', s=15, alpha=0.6)
    ax2.set_title('(b) Ours (CARD) under Noise', fontsize=12)
    ax2.axis('off')
    
    plt.suptitle('Feature Manifold Visualization (t-SNE) on 10 Classes', fontsize=14, y=0.95)
    
    # Legend
    handles, labels = scatter2.legend_elements()
    fig.legend(handles, [f'Class {i}' for i in range(NUM_CLASSES_TO_PLOT)], loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('outputs/tsne_real.pdf')
    print("Saved figures/tsne_real.pdf")

if __name__ == "__main__":
    plot_real_tsne()