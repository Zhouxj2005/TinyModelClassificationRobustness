# train_distill.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import csv
import time
from models import get_model
from augmix_ops import AugMixDataset

# --- 参数配置 ---
parser = argparse.ArgumentParser(description='Robust Distillation Training')
parser.add_argument('--student', default='mobilenet_v2', type=str, help='Student model architecture')
parser.add_argument('--teacher', default='resnet18_cifar', type=str, help='Teacher model architecture')
parser.add_argument('--teacher_path', default='./final_checkpoint/resnet18-sota.pth', type=str, help='Path to teacher checkpoint')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.05, type=float) # 蒸馏通常可以用稍微大一点的LR
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--alpha', default=15.0, type=float, help='Weight for KD Loss')
parser.add_argument('--temperature', default=3.0, type=float, help='Temperature for KD')
parser.add_argument('--save_dir', default='./checkpoint', type=str)
args = parser.parse_args()

# --- 核心组件：KL 散度蒸馏 Loss ---
def loss_kd(outputs, teacher_outputs, temperature):
    """
    Args:
        outputs: Student 的 Logits
        teacher_outputs: Teacher 的 Logits
        temperature: 软化温度 (T越大，分布越平滑，关注非主类信息)
    """
    T = temperature
    # KLDivLoss 期望输入是 log_softmax，目标是 softmax
    loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1)
    ) * (T * T) # 按照 Hinton 的论文，需要乘 T^2 保持梯度量级
    return loss

def train_distill():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始鲁棒蒸馏 (Robust Distillation) ===")
    print(f"   Teacher: {args.teacher} | Student: {args.student}")
    print(f"   Alpha: {args.alpha} | Temperature: {args.temperature}")

    # 1. 准备数据 (和 AugMix 训练一致)
    transform_final = transforms.Compose([transforms.ToTensor()])
    transform_train_base = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    
    trainset_raw = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train_base)
    trainset = AugMixDataset(trainset_raw, transform_final)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 加载 Teacher 模型 (冻结参数)
    print(f"=> Loading Teacher from {args.teacher_path}...")
    teacher = get_model(args.teacher, num_classes=100).to(device)
    # 加载权重
    try:
        checkpoint = torch.load(args.teacher_path)
        teacher.load_state_dict(checkpoint['net'])
        print(f"   Teacher Accuracy (Recorded): {checkpoint.get('acc', 'N/A')}%")
    except Exception as e:
        print(f"❌ Error loading teacher: {e}")
        return

    teacher.eval() # ⚠️ 关键：Teacher 必须始终处于 Eval 模式
    # 冻结参数，节省显存和计算
    for param in teacher.parameters():
        param.requires_grad = False

    # 3. 初始化 Student 模型
    student = get_model(args.student, num_classes=100).to(device)

    # 4. 优化器
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 日志
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    log_path = os.path.join(args.save_dir, f'{args.student}_distill_{args.alpha}_{args.epochs}_{args.lr}_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'clean_acc', 'time'])

    best_acc = 0.0

    # 5. 训练循环
    for epoch in range(args.epochs):
        student.train()
        train_loss = 0.0
        start_time = time.time()
        
        for i, (images_clean, images_aug1, images_aug2, targets) in enumerate(trainloader):
            # Move to GPU
            images_clean, images_aug1, images_aug2, targets = \
                images_clean.to(device), images_aug1.to(device), images_aug2.to(device), targets.to(device)
            
            # 拼接: Batch x 3
            images_all = torch.cat([images_clean, images_aug1, images_aug2], dim=0)

            # --- Forward Pass ---
            optimizer.zero_grad()

            # 1. Student Forward
            logits_all_s = student(images_all)
            logits_clean_s, logits_aug1_s, logits_aug2_s = torch.split(logits_all_s, images_clean.size(0))

            # 2. Teacher Forward (No Grad)
            with torch.no_grad():
                logits_all_t = teacher(images_all)
                # 我们不需要拆分 Teacher 的 logits，因为我们是整体蒸馏

            # --- Loss Calculation (你的创新组合拳) ---
            
            # Part A: Cross Entropy (Student 必须做对 Clean 样本的分类)
            loss_ce = F.cross_entropy(logits_clean_s, targets)

            # Part B: AugMix Consistency (Student 自我约束：不同视角输出要一致)
            p_clean = F.softmax(logits_clean_s, dim=1)
            p_aug1 = F.softmax(logits_aug1_s, dim=1)
            p_aug2 = F.softmax(logits_aug2_s, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_js = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            # Part C: Knowledge Distillation (Teacher 约束：你要像我一样思考)
            # 关键：我们在所有样本(Clean + Aug)上都进行蒸馏
            loss_kd_val = loss_kd(logits_all_s, logits_all_t, args.temperature)

            # --- Total Loss ---
            # 这里的系数是可以调优的超参数
            # 1.0 * CE + 12 * JS + 1.0 * KD
            loss = loss_ce + 12 * loss_js + args.alpha * loss_kd_val

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        student.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100.*correct/total
        scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = train_loss/(i+1)
        print(f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | Loss: {avg_loss:.3f} | loss_ce: {loss_ce} | loss_js: {12*loss_js} | loss_kd_val: {args.alpha * loss_kd_val} | Clean Acc: {acc:.2f}%")
        
        # Log & Save
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, acc, epoch_time])

        if acc > best_acc:
            best_acc = acc
            torch.save({'net': student.state_dict(), 'acc': acc}, 
                       os.path.join(args.save_dir, f'{args.student}_distill_{args.alpha}_{args.epochs}_{args.lr}.pth'))

    print(f"=== 蒸馏完成. Best Acc: {best_acc:.2f}% ===")

if __name__ == "__main__":
    train_distill()
```

# train_distill_improved.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import csv
import time
import math
from models import get_model
from augmix_ops import AugMixDataset

# ==========================================
# 0. 参数配置
# ==========================================
parser = argparse.ArgumentParser(description='Improved Robust Distillation Training')
parser.add_argument('--student', default='mobilenet_v2', type=str)
parser.add_argument('--teacher', default='resnet18', type=str)
parser.add_argument('--teacher_path', default='./checkpoint/resnet18_200_0.1_augmix.pth', type=str)
parser.add_argument('--epochs', default=200, type=int)
# MobileNetV2 配合 Warmup 建议使用 0.1，如果不稳可降至 0.05
parser.add_argument('--lr', default=0.1, type=float) 
parser.add_argument('--batch_size', default=128, type=int)

# --- 蒸馏超参数 ---
parser.add_argument('--alpha', default=20.0, type=float, help='Weight for Soft Target KD')
parser.add_argument('--beta', default=500.0, type=float, help='Weight for RKD (Structure) Loss')
parser.add_argument('--temperature', default=4.0, type=float, help='Temperature for KD')
parser.add_argument('--js_lambda', default=12.0, type=float, help='Weight for AugMix Consistency')

parser.add_argument('--save_dir', default='./checkpoint', type=str)
args = parser.parse_args()

# ==========================================
# 1. 工具类: FeatureWrapper
#    (无需修改 models.py 即可提取特征)
# ==========================================
class FeatureWrapper(nn.Module):
    """
    包装器：通过 Hook 机制自动提取倒数第二层的特征。
    返回: (logits, features)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook_handle = None
        self._register_hook()

    def _find_target_layer(self):
        # 递归寻找真正的模型实体（处理 NormalizedModel 包装）
        real_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        # 针对不同架构寻找“最后一层全连接层”
        if hasattr(real_model, 'linear'): 
            return real_model.linear # ResNetRobustBench
        elif hasattr(real_model, 'fc'):
            return real_model.fc # Standard ResNet
        elif hasattr(real_model, 'classifier'):
            # MobileNetV2: classifier 是一个 Sequential
            return real_model.classifier 
        else:
            raise ValueError("无法自动定位最后一层 (Linear/FC/Classifier)，请检查模型结构。")

    def _hook_fn(self, module, input):
        # 🌟 修正点：pre_hook 只有 (module, input) 两个参数，没有 output
        # 全连接层的输入就是我们要的特征
        # input 是一个 tuple，取第一个元素
        feat = input[0]
        # 展平特征 [Batch, C, 1, 1] -> [Batch, C]
        self.features = feat.flatten(1)

    def _register_hook(self):
        target_layer = self._find_target_layer()
        # 注册 Forward Pre Hook：在进入全连接层之前截获输入
        # 这里的输入就是 Feature
        self.hook_handle = target_layer.register_forward_pre_hook(self._hook_fn)

    def forward(self, x):
        # 这一步会触发 hook，更新 self.features
        logits = self.model(x)
        return logits, self.features

# ==========================================
# 2. 损失函数 (Loss Functions)
# ==========================================

def rkd_loss(student_feat, teacher_feat):
    """
    Relational Knowledge Distillation (Distance-wise)
    让 Student 学习 Teacher 的样本间几何距离关系。
    """
    # 1. 特征归一化 (消除量纲差异)
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    # 2. 计算成对欧氏距离矩阵 [Batch, Batch]
    t_dist = torch.cdist(teacher_feat, teacher_feat, p=2)
    s_dist = torch.cdist(student_feat, student_feat, p=2)

    # 3. 归一化距离矩阵 (除以矩阵均值，关注相对关系而非绝对数值)
    # 加上 epsilon 防止除零
    t_mean = t_dist.mean() + 1e-8
    s_mean = s_dist.mean() + 1e-8
    
    t_dist_norm = t_dist / t_mean
    s_dist_norm = s_dist / s_mean

    # 4. 计算矩阵差异 (Huber Loss 比 MSE 更稳健)
    loss = F.smooth_l1_loss(s_dist_norm, t_dist_norm)
    return loss

def confidence_weighted_kd_loss(outputs, teacher_outputs, temperature):
    """
    置信度加权的 KD Loss。
    Teacher 越确定 (Max Prob 越高)，Loss 权重越大。
    """
    T = temperature
    
    # 1. 计算 Teacher 的置信度权重
    with torch.no_grad():
        t_probs = F.softmax(teacher_outputs, dim=1)
        t_conf, _ = t_probs.max(dim=1) # [Batch]
        # 权重可以直接用置信度，也可以做一个非线性映射
        loss_weight = t_conf.detach()

    # 2. 计算逐样本的 KL 散度
    # reduction='none' 保留 [Batch] 维度
    loss_pointwise = nn.KLDivLoss(reduction='none')(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1)
    ) * (T * T)
    
    # KLDivLoss 输出通常是 [Batch, Classes]，求和得到每个样本的 Loss
    loss_sample = loss_pointwise.sum(dim=1)
    
    # 3. 加权平均
    loss = (loss_sample * loss_weight).mean()
    return loss

def get_lr_scheduler(optimizer, total_epochs, warmup_epochs=5):
    """ Warmup + Cosine Scheduler """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================================
# 3. 主训练流程
# ==========================================
def train_distill_improved():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始改进版鲁棒蒸馏 (Improved Robust Distillation) ===")
    print(f"   Teacher: {args.teacher} -> Student: {args.student}")
    print(f"   KD Alpha: {args.alpha} | RKD Beta: {args.beta} | JS Lambda: {args.js_lambda}")

    # --- 1. 数据准备 ---
    # 这里的 transform 只做 ToTensor，归一化由模型内部 NormalizedModel 完成
    transform_final = transforms.Compose([transforms.ToTensor()])
    transform_train_base = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    
    trainset_raw = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train_base)
    trainset = AugMixDataset(trainset_raw, transform_final)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # --- 2. 模型加载与包装 ---
    print(f"=> Preparing models...")
    
    # Teacher
    teacher_raw = get_model(args.teacher, num_classes=100).to(device)
    # 加载权重逻辑
    if os.path.exists(args.teacher_path):
        ckpt = torch.load(args.teacher_path, map_location=device)
        # 兼容处理
        if isinstance(ckpt, dict) and 'net' in ckpt: state_dict = ckpt['net']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        else: state_dict = ckpt
        
        # 移除 model. 前缀 (如果有) 因为 teacher_raw 此时还没 wrap
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        msg = teacher_raw.load_state_dict(state_dict, strict=False)
        print(f"   Teacher loaded: {msg}")
    else:
        print(f"❌ Error: Teacher path not found {args.teacher_path}")
        return

    teacher_raw.eval()
    for p in teacher_raw.parameters(): p.requires_grad = False
    
    # 🌟 使用 FeatureWrapper 包装 Teacher
    teacher = FeatureWrapper(teacher_raw)

    # Student
    student_raw = get_model(args.student, num_classes=100).to(device)
    # 🌟 使用 FeatureWrapper 包装 Student
    student = FeatureWrapper(student_raw)

    # --- 3. 优化器 ---
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = get_lr_scheduler(optimizer, args.epochs, warmup_epochs=5)

    # --- 4. 日志 ---
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    log_path = os.path.join(args.save_dir, f'{args.student}_{args.teacher}_distill_improved_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'loss_ce', 'loss_js', 'loss_kd', 'loss_rkd', 'clean_acc', 'time'])

    best_acc = 0.0

    # --- 5. 训练循环 ---
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        start_time = time.time()
        
        # 记录分项 Loss 方便 debug
        m_ce, m_js, m_kd, m_rkd = 0.0, 0.0, 0.0, 0.0
        
        for i, (images_clean, images_aug1, images_aug2, targets) in enumerate(trainloader):
            images_clean, images_aug1, images_aug2, targets = \
                images_clean.to(device), images_aug1.to(device), images_aug2.to(device), targets.to(device)
            
            # 拼接
            images_all = torch.cat([images_clean, images_aug1, images_aug2], dim=0)

            optimizer.zero_grad()

            # --- Forward ---
            # Student: 获取 logits 和 features
            logits_all_s, feats_all_s = student(images_all)
            logits_clean_s, logits_aug1_s, logits_aug2_s = torch.split(logits_all_s, images_clean.size(0))
            # 只取 Clean 数据的特征做 RKD (避免 AugMix 的强扭曲破坏流形结构)
            feats_clean_s, _, _ = torch.split(feats_all_s, images_clean.size(0))

            # Teacher: Forward (No Grad)
            with torch.no_grad():
                logits_all_t, feats_all_t = teacher(images_all)
                feats_clean_t, _, _ = torch.split(feats_all_t, images_clean.size(0))

            # --- Loss Calculation ---

            # 1. CE Loss (Clean Classification)
            loss_ce = F.cross_entropy(logits_clean_s, targets)

            # 2. AugMix JS Consistency
            p_clean = F.softmax(logits_clean_s, dim=1)
            p_aug1 = F.softmax(logits_aug1_s, dim=1)
            p_aug2 = F.softmax(logits_aug2_s, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_js = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            # 3. Weighted KD Loss (Clean + Aug)
            # 使用改进的 "Confidence-Aware" KD
            loss_kd_val = confidence_weighted_kd_loss(logits_all_s, logits_all_t, args.temperature)

            # 4. RKD Loss (Clean Structure)
            # 改进的 "Relational" 蒸馏
            loss_rkd_val = rkd_loss(feats_clean_s, feats_clean_t)

            # --- Total Loss ---
            loss = loss_ce + \
                   args.js_lambda * loss_js + \
                   args.alpha * loss_kd_val + \
                   args.beta * loss_rkd_val

            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            m_ce += loss_ce.item()
            m_js += loss_js.item()
            m_kd += loss_kd_val.item()
            m_rkd += loss_rkd_val.item()

        # Validation
        student.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                # 注意：student 现在是 FeatureWrapper，输出是 tuple
                outputs, _ = student(inputs) 
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100.*correct/total
        scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / (i+1)
        
        # 打印详细 Loss 组成，方便观察哪个 Loss 在起作用
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.0f}s | Acc: {acc:.2f}%")
        print(f"   Loss: {avg_loss:.3f} (CE:{m_ce/(i+1):.2f} JS:{m_js/(i+1):.2f} KD:{m_kd/(i+1):.2f} RKD:{m_rkd/(i+1):.4f})")
        
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, m_ce/(i+1), m_js/(i+1), m_kd/(i+1), m_rkd/(i+1), acc, epoch_time])

        if acc > best_acc:
            best_acc = acc
            # 保存时，我们保存 student.model 的 state_dict
            # 这样以后加载就不需要 FeatureWrapper 了，变回普通的 NormalizedModel
            torch.save({'net': student.model.state_dict(), 'acc': acc}, 
                       os.path.join(args.save_dir, f'{args.student}_{args.teacher}_distill_improved_best.pth'))

    print(f"=== 训练完成. Best Clean Acc: {best_acc:.2f}% ===")

if __name__ == "__main__":
    train_distill_improved()
```