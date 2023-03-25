import argparse
import os

import numpy as np
import torch
from sklearn import preprocessing
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
    x = x.to(device)
    y = y.to(device)

    with autocast():
        prob = model.fc(x)
        loss = criterion(prob, y)

    return loss, prob, y


def get_features_from_encoder(model, loader, device):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = torch.cat(model.attention_forward(x.to(device)), dim=1).to(device)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader


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

    model = DualAttentionNet(pretrained=False, num_classes=10).to(device)
    model.load_state_dict(torch.load('OUR50.pth'), strict=False)
    for name, param in model.named_parameters():
        if name in 'backbone':
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0003, weight_decay=0.0008)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    loss_scaler = utils.NativeScalerWithGradUpdate()
    criterion = nn.CrossEntropyLoss()

    # Generate Features
    model.eval()
    x_train, y_train = get_features_from_encoder(model, train_loader, device)
    x_test, y_test = get_features_from_encoder(model, test_loader, device)

    if len(x_train.shape) > 2:
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train.detach().cpu())
    x_train = scaler.transform(x_train.detach().cpu()).astype(np.float32)
    x_test = scaler.transform(x_test.detach().cpu()).astype(np.float32)
    train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train,
                                                                torch.from_numpy(x_test), y_test)

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
