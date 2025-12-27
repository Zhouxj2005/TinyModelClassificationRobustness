import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. RobustBench 风格的 ResNet18 定义
#    (完全匹配 Modas2021PRIMEResNet18 的结构)
# ==========================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 注意：这里必须叫 shortcut，不能叫 downsample
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetRobustBench(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNetRobustBench, self).__init__()
        self.in_planes = 64

        # --- 核心：定义归一化参数 ---
        # RobustBench 的模型通常自带这两个参数。
        # 这里先注册 buffer，加载权重时会被 .pth 文件中的真实值覆盖。
        self.register_buffer('mu', torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 注意：必须叫 linear，不能叫 fc
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 标准化 (匹配 RobustBench 的处理逻辑)
        x = (x - self.mu) / self.sigma
        
        # 2. ResNet 主体
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        feat = out.view(out.size(0), -1) # [Batch, 512] -> 这是我们要的特征
        logits = self.linear(feat)
        return logits, feat  # <--- 返回 Tuple

# 针对 MobileNetV2 (Student) 的修改
# 我们通常需要重写一下 torchvision 的 mobilenet forward
class MobileNetV2_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV2_CIFAR, self).__init__()
        self.net = models.mobilenet_v2(pretrained=False)
        # 修改第一层
        self.net.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改分类器
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, num_classes)

    def forward(self, x):
        # 提取特征
        out = self.net.features(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        feat = torch.flatten(out, 1) # [Batch, 1280]
        logits = self.net.classifier(feat)
        return logits, feat # <--- 返回 Tuple
# ==========================================
# 2. 模型工厂
# ==========================================
class NormalizedModel(nn.Module):
    """
    包装器：在模型前加入归一化层。
    使得模型可以像 RobustBench 模型一样接收 [0, 1] 的输入。
    """
    def __init__(self, model):
        super(NormalizedModel, self).__init__()
        self.model = model
        # CIFAR-100 均值和方差
        self.register_buffer('mu', torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.model(x)


def get_model(name, num_classes=100):
    """
    模型工厂：获取适配 CIFAR-100 (32x32) 的模型
    """
    print(f"=> 创建模型: {name} (适配 CIFAR-100)...")

    if name == 'resnet18_cifar':
        # Teacher: 自带归一化，直接返回
        return ResNetRobustBench(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
    # --- 以下模型是 Student，通常不带归一化，我们需要包装它 ---
    if name == 'resnet18':
        # 加载标准 ResNet18
        model = models.resnet18(pretrained=False)
        
        # 1. 修改第一层卷积：适应 32x32 输入，不进行下采样
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 去掉 MaxPool：小图不需要一开始就丢弃信息
        model.maxpool = nn.Identity()
        
        # 3. 修改全连接层：适应 100 类
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif name == 'mobilenet_v2':
        model = MobileNetV2_CIFAR(num_classes)
        return NormalizedModel(model) # 包装
        
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

    return NormalizedModel(model)

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