import argparse
import os

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10

from dataset import get_transform
from network.dual_attention_net import DualAttentionNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def one_iter(data, model, criterion, device):
    x, y = data
    x = x.to(memory_format=torch.channels_last).to(device)
    y = y.to(device)

    with autocast():
        pixel_attn, channel_attn = model.attention_forward(x)

    return pixel_attn, channel_attn, y


def main_worker():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=100,
                        type=int)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    basic_transform, _, _ = get_transform()

    dataset_train = CIFAR10('/data/image_classification', train=True, transform=basic_transform)
    dataset_test = CIFAR10('/data/image_classification', train=False, transform=basic_transform)
    train_loader = DataLoader(dataset_train, batch_size=2048, num_workers=4, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=2048, num_workers=4, shuffle=False)

    model = DualAttentionNet(pretrained=False, num_classes=10).to(memory_format=torch.channels_last).to(device)
    model.load_state_dict(torch.load('OUR10.pth'), strict=False)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    top1 = Accuracy(compute_on_step=False)
    top5 = Accuracy(compute_on_step=False, top_k=5)

    classes = torch.tensor(range(10))
    gallery = torch.zeros((10, 2, 256), dtype=torch.half)

    for batch_idx, data in enumerate(train_loader):
        pixel_attn, channel_attn, target = one_iter(data, model, criterion, device)
        features = torch.stack([pixel_attn, channel_attn], dim=1).detach().cpu()

        for c in classes:
            idx = torch.where(target == c)
            if len(idx[0]) != 0:
                gallery[c] = features[idx].mean(0)

    for batch_idx, data in enumerate(test_loader):
        pixel_attn, channel_attn, target = one_iter(data, model, criterion, device)
        features = torch.stack([pixel_attn, channel_attn], dim=1).detach().cpu()

        b = features.size(0)

        similarity = -(gallery.unsqueeze(0) - features.unsqueeze(1)).norm(dim=2).pow(2).view(b, 10, -1).sum(-1)

        top1.update(similarity, target.detach().cpu())
        top5.update(similarity, target.detach().cpu())

    print(f'Top1: {top1.compute()} | Top5: {top5.compute()}')


if __name__ == '__main__':
    main_worker()
