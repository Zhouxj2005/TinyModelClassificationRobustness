import torch
import torch.nn as nn
import torchvision.models as models

def get_model(name, num_classes=100):
    """
    模型工厂：获取适配 CIFAR-100 (32x32) 的模型
    """
    print(f"=> 创建模型: {name} (适配 CIFAR-100)...")
    
    if name == 'resnet18':
        # 加载标准 ResNet18
        model = models.resnet18(pretrained=False)
        
        # --- 关键修改开始 ---
        # 1. 修改第一层卷积：适应 32x32 输入，不进行下采样
        # 原版: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 去掉 MaxPool：小图不需要一开始就丢弃信息
        # 原版: MaxPool2d(kernel_size=3, stride=2, padding=1)
        model.maxpool = nn.Identity()
        
        # 3. 修改全连接层：适应 100 类
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # --- 关键修改结束 ---
        
    elif name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        
        # MobileNetV2 的第一层在 features[0][0]
        # 原版: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 修改为 stride=1，保留分辨率
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改分类器
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(pretrained=False)
        # ShuffleNet 的第一层
        model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"未支持的模型: {name}")

    return model

# --- 测试模型结构 ---
if __name__ == "__main__":
    # 模拟一张 CIFAR 图片 (Batch=2, Channel=3, Height=32, Width=32)
    x = torch.randn(2, 3, 32, 32)
    
    for net_name in ['resnet18', 'mobilenet_v2']:
        net = get_model(net_name, num_classes=100)
        y = net(x)
        print(f"[{net_name}] 输出形状: {y.shape}") 
        # 预期输出: torch.Size([2, 100])
        # 如果没有修改好，可能会报错或者输出形状不对