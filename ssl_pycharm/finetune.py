import argparse
import os

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from torchvision.datasets import CIFAR10

import utils
from dataset import get_transform
from network.dual_attention_net import DualAttentionNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def one_iter(data, model, criterion, device):
    x, y = data
    x = x.to(memory_format=torch.channels_last).to(device)
    y = y.to(device)

    with autocast():
        prob = model.regression(x)
        loss = criterion(prob, y)

    return loss, prob, y


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
    model.load_state_dict(torch.load('2OUR50.pth'), strict=False)
    for name, param in model.named_parameters():
        if name in 'backbone':
            param.requires_grad = False
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0008)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    loss_scaler = utils.NativeScalerWithGradUpdate()
    criterion = nn.CrossEntropyLoss()

    losses = MeanMetric(compute_on_step=False).to(device)
    top1 = Accuracy(compute_on_step=False).to(device)
    top5 = Accuracy(compute_on_step=False, top_k=5).to(device)

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        model.train()
        for batch_idx, data in enumerate(train_loader):
            loss, prob, target = one_iter(data, model, criterion, device)
            loss_scaler(loss, optimizer, clip_grad=5.0, parameters=model.fc.parameters())
            optimizer.zero_grad()

            losses.update(loss.item() / prob.size(0))
            top1.update(prob, target)
            top5.update(prob, target)

        print(
            f'Train[{epoch:>2}] loss: {losses.compute():.3f}, top1:{top1.compute() * 100}, top5:{top5.compute() * 100}')
        top1.reset()
        top5.reset()
        losses.reset()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            # torch.save(model.state_dict(), f'F_OUR{epoch + 1}.pth')
            model.eval()
            for batch_idx, data in enumerate(test_loader):
                loss, prob, target = one_iter(data, model, criterion, device)
                losses.update(loss.item() / prob.size(0))
                top1.update(prob, target)
                top5.update(prob, target)

            print(
                f'===Valid[{epoch:>2}] loss: {losses.compute():.3f}, top1:{top1.compute() * 100}, top5:{top5.compute() * 100}')
            top1.reset()
            top5.reset()
            losses.reset()


if __name__ == '__main__':
    main_worker()
