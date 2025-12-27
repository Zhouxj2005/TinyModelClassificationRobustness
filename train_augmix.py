import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
import csv
from models import get_model
from augmix_ops import AugMixDataset # 导入刚才写的

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--model', default='mobilenet_v2', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--save_dir', default='./checkpoint', type=str)
args = parser.parse_args()

def train_augmix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始 AugMix 训练: {args.model} ===")

    # 1. 数据准备
    
    
    # 基础预处理 (最后一步)
    transform_final = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 基础增强 (给 Clean 样本用的)
    transform_train_base = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    # 加载原始 CIFAR100
    # 注意：这里 dataset 的 transform 只做基础增强，不做 ToTensor/Normalize
    # 因为 AugMix 需要操作 PIL Image
    trainset_raw = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train_base)
    
    # 包装成 AugMix Dataset
    trainset = AugMixDataset(trainset_raw, transform_final)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16,pin_memory=True)

    # 测试集保持不变
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16,pin_memory=True)

    # 2. 模型与优化器
    net = get_model(args.model, num_classes=100).to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.pretrained:
        if not os.path.exists(args.pretrained):
            print(f"Error: 找不到预训练权重文件 {args.pretrained}")
            return
        checkpoint = torch.load(args.pretrained)
        net.load_state_dict(checkpoint['net'])
        print(f"Loaded Pretrained Weights from {args.pretrained}")

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    
    best_acc = 0.0

    # --- 📝 新增: 初始化日志文件 ---
    log_path = os.path.join(args.save_dir, f'{args.model}_{args.epochs}_{args.lr}_augmix_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['epoch', 'loss', 'clean_acc', 'time'])
    print(f"日志将保存至: {log_path}")
    
    # 3. 训练循环
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0.0
        start_time = time.time()
        
        for i, (images_clean, images_aug1, images_aug2, targets) in enumerate(trainloader):
            images_clean, images_aug1, images_aug2, targets = \
                images_clean.to(device), images_aug1.to(device), images_aug2.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # --- AugMix 核心逻辑 ---
            # 为了节省显存，通常把三组图拼在一起过一次模型 (Batch x 3)
            # 1. 把图片拼接起来: [Batch*3, 3, 32, 32]
            images_all = torch.cat([images_clean, images_aug1, images_aug2], dim=0)

            # 2. 一次性过模型
            logits_all = net(images_all)

            # 3. 再拆分回来
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images_clean.size(0))

            loss_clean = F.cross_entropy(logits_clean, targets)
            
            # 4. Jensen-Shannon Consistency Loss
            p_clean = F.softmax(logits_clean, dim=1)
            p_aug1 = F.softmax(logits_aug1, dim=1)
            p_aug2 = F.softmax(logits_aug2, dim=1)
            
            # 混合分布 M = (P + P_aug1 + P_aug2) / 3
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            
            # JS Div = KL(P||M) + KL(P1||M) + KL(P2||M)
            loss_js = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            
            # 总 Loss: Classification + 12 * JS_Consistency
            loss = loss_clean + 12 * loss_js
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        net.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100.*correct/total
        scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = train_loss/(i+1)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | Loss: {avg_loss:.3f} | Clean Acc: {acc:.2f}%")

        # --- 📝 新增: 写入日志 ---
        # 使用 'a' (append) 模式，防止程序中途崩溃数据丢失
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, acc, epoch_time])
        # -----------------------
        
        if acc > best_acc:
            best_acc = acc
            torch.save({'net': net.state_dict(), 'acc': acc}, 
                       os.path.join(args.save_dir, f'{args.model}_{args.epochs}_{args.lr}_augmix.pth'))

    print(f"=== AugMix 训练结束. Best Clean Acc: {best_acc:.2f}% ===")

if __name__ == "__main__":
    train_augmix()