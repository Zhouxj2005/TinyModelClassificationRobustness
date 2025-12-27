import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# CIFAR-100 的增强操作集 (不包含 Corruptions 中的测试操作，防止作弊)
def int_parameter(level, maxval):
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def augmentations(pil_img, severity=3, width=3, depth=-1, alpha=1.):
    """
    AugMix 的核心逻辑
    Args:
        pil_img: 输入图像
        severity: 增强强度 (1-10)
        width: 混合链的宽度 (通常为3)
        depth: 链的深度 (通常随机)
        alpha: Beta分布参数
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(np.array(pil_img), dtype=np.float32)

    for i in range(width):
        image_aug = pil_img.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        
        for _ in range(d):
            op = np.random.choice(augmentations_list)
            image_aug = op(image_aug, severity)
            
        # 混合
        mix += ws[i] * np.array(image_aug, dtype=np.float32)

    mixed = (1 - m) * np.array(pil_img, dtype=np.float32) + m * mix
    return Image.fromarray(np.uint8(mixed))

# --- 具体的增强算子 (Primitive Operations) ---
def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.random() > 0.5: degrees = -degrees
    return pil_img.rotate(degrees)

def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.random() > 0.5: level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.random() > 0.5: level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5: level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5: level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

# 操作列表
augmentations_list = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y
]

class AugMixDataset(object):
    """
    一个 Wrapper，用于 PyTorch Dataset。
    它会返回 3 张图: (clean, aug1, aug2)
    """
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess # 这里的 preprocess 包含 ToTensor 和 Normalize

    def __getitem__(self, i):
        x, y = self.dataset[i] # x 是 PIL Image
        
        # AugMix 逻辑：生成两张不同的增强图
        x_aug1 = augmentations(x)
        x_aug2 = augmentations(x)
        
        return self.preprocess(x), self.preprocess(x_aug1), self.preprocess(x_aug2), y

    def __len__(self):
        return len(self.dataset)