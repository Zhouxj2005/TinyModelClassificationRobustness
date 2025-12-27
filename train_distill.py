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