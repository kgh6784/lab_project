import argparse
import os

import torch
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

import utils
from dataset import get_transform, MyCIFAR10
from network.dual_attention_net import DualAttentionNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use


def one_iter(data, model, device):
    x, x_pixel, x_channels = data
    x = x.to(memory_format=torch.channels_last).to(device)
    x_pixel = x_pixel.to(memory_format=torch.channels_last).to(device)
    x_channels = x_channels.to(memory_format=torch.channels_last).to(device)
    with autocast():
        loss, sim1, sim2 = model(x, x_pixel, x_channels)
    return loss, sim1, sim2


def main_worker():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=50,
                        type=int)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    basic_transform, pixel_transform, channel_transform = get_transform()

    # Create train dataloader
    dataset_train = MyCIFAR10('/data/image_classification', True, basic_transform, pixel_transform, channel_transform)
    train_loader = DataLoader(dataset_train, batch_size=2048, num_workers=4, shuffle=True)

    model = DualAttentionNet(pretrained=False).to(memory_format=torch.channels_last).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    loss_scaler = utils.NativeScalerWithGradUpdate()

    sim1s = MeanMetric(compute_on_step=False).to(device)
    sim2s = MeanMetric(compute_on_step=False).to(device)

    loader_length = len(train_loader)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_loader):
            loss, sim1, sim2 = one_iter(data, model, device)
            loss_scaler(loss, optimizer, parameters=model.parameters())
            optimizer.zero_grad()
            scheduler.step()

            sim1s.update(sim1)
            sim2s.update(sim2)

            if batch_idx % 20 == 0 or batch_idx == loader_length - 1:
                print(f"[Train[{epoch:>2}] {batch_idx + 1:>3}/{loader_length} loss: {loss.item():.6f}")

        print(f'[{epoch:>2}] sim_pix:{sim1s.compute()}, sim_cha:{sim2s.compute()}')
        sim1s.reset()
        sim2s.reset()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'2OUR{epoch + 1}.pth')


if __name__ == '__main__':
    main_worker()
