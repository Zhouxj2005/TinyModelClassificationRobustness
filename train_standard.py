import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
import csv
from models import get_model # 导入咱们刚才写的模型工厂

# --- 配置参数 ---
parser = argparse.ArgumentParser(description='Standard Training on CIFAR-100')
parser.add_argument('--model', default='mobilenet_v2', type=str, help='model name: mobilenet_v2 | resnet18')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--save_dir', default='./checkpoint', type=str, help='directory to save checkpoints')
args = parser.parse_args()

def train():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始训练: {args.model} | Device: {device} ===")

    # 2. 数据准备 (Standard Augmentation)
    print("=> 准备数据...")
    # CIFAR-100 官方统计的均值和方差
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. 模型构建
    net = get_model(args.model, num_classes=100)
    net = net.to(device)

    # 4. 优化器与Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 使用余弦退火学习率，训练后期收敛更稳
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = os.path.join(args.save_dir, f'{args.model}_{args.epochs}_{args.lr}_standard_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['epoch', 'loss', 'clean_acc', 'time'])
    print(f"日志将保存至: {log_path}")
    
    # 5. 训练循环
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    best_acc = 0.0

    for epoch in range(args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 验证阶段 (只测 Clean Accuracy)
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        scheduler.step() # 更新学习率

        avg_loss = train_loss / (batch_idx + 1)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | "
              f"Loss: {train_loss/(batch_idx+1):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Test Acc: {acc:.2f}%")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, acc, epoch_time])
        # -----------------------
        
        # 保存最佳模型
        if acc > best_acc:
            print(f"Found new best model! ({best_acc:.2f}% -> {acc:.2f}%) Saving...")
            best_acc = acc
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            save_path = os.path.join(args.save_dir, f'{args.model}_{args.epochs}_{args.lr}_standard.pth')
            torch.save(state, save_path)

    print(f"=== 训练结束. Best Clean Accuracy: {best_acc:.2f}% ===")

if __name__ == "__main__":
    train()