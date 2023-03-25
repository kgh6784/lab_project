import os.path
from glob import glob

import torch
from PIL import Image
from matplotlib import pyplot as plt
from monai.visualize import GradCAM
from torchvision.transforms import transforms

from dataset import get_transform
from network.byol.models.resnet_base_network import BYOLResNet18
from network.dual_attention_net2 import DualAttentionNet
from network.simclr import ResNetSimCLR

imgs = glob('sample_images/*')
if not os.path.exists('grad_images'):
    os.mkdir('grad_images')
print(imgs)

DAN = DualAttentionNet(num_classes=10)
DAN.load_state_dict(torch.load('OUR50.pth'), strict=False)

# load pre-trained parameters
byol = BYOLResNet18()
load_params = torch.load(os.path.join('network/byol/runs/Dec06_09-59-30_CIDock3/checkpoints/model.pth'))

if 'online_network_state_dict' in load_params:
    byol.load_state_dict(load_params['online_network_state_dict'], strict=False)

simCLR = ResNetSimCLR(out_dim=10)
state_dict = torch.load('network/simclr/checkpoint_0100.pth.tar')['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
            state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]
simCLR.backbone.load_state_dict(state_dict, strict=False)

models = [
    ('DAN', DAN, ['pixel_attention.0', 'channel_attention.0']),
    ('byol', byol, 'encoder.7'),
    ('simCLR', simCLR, 'backbone.layer4'),
]

size = 96
basic_aug, pos_aug, channel_aug = get_transform(size)
resize = round(size * 1.13)
resize_aug = transforms.Compose([transforms.Resize((resize, resize)),
                                 transforms.CenterCrop((size, size))])

for i, img in enumerate(imgs):
    img = Image.open(img)
    aug_img = basic_aug(img)
    resize_img = resize_aug(img)
    for model_info in models:
        model_name, model, target_layers = model_info
        if isinstance(target_layers, list):
            cam1 = \
                GradCAM(nn_module=model, target_layers=target_layers[0], postprocessing=None)(x=aug_img.unsqueeze(0))[0]
            cam2 = \
                GradCAM(nn_module=model, target_layers=target_layers[1], postprocessing=None)(x=aug_img.unsqueeze(0))[0]
            cam = torch.cat([cam1, cam2], dim=0).sum(0).unsqueeze(0)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
        else:
            cam = GradCAM(nn_module=model, target_layers=target_layers, postprocessing=None)(x=aug_img.unsqueeze(0))[0]
        plt.imshow(resize_img)
        plt.imshow(cam[0].detach().cpu().numpy(), alpha=0.5, cmap='jet')
        plt.axis('off')  # x,y축 모두 없애기
        plt.tight_layout()
        plt.savefig(f'grad_images/{i}_{model_name}.png')
        # plt.show()
