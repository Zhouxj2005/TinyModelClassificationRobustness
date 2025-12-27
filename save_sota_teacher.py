from robustbench.utils import load_model
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
import torch

model = load_model(
    model_name='Modas2021PRIMEResNet18',
    dataset=BenchmarkDataset.cifar_100,
    threat_model=ThreatModel.corruptions
)
state = {
    'net': model.state_dict(),
    'acc': None,          # teacher 一般不需要 clean acc
    'epoch': -1,          # 表示非你自己训练的
    'source': 'RobustBench Modas2021PRIME'
}

save_path = './resnet18-sota.pth'
torch.save(state, save_path)

print(f"Teacher model saved to {save_path}")