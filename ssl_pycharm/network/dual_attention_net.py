import torch
from timm import create_model
from torch import nn

from network.attention_module.attention import ChannelAttention, PixelAttention


class AttBlock(nn.Module):
    def __init__(self, dim, attn_fn, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = attn_fn(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = x + self.attn(self.norm(x.transpose(1, 3)).transpose(1, 3))
        return self.act(x)


class DualAttentionNet(nn.Module):
    def __init__(self, backbone='resnet18', dim=512, pretrained=False, num_classes=None):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=pretrained)
        self.backbone.global_pool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.channel_attention = nn.Sequential(
            AttBlock(dim, ChannelAttention),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, dim // 2)
        )
        self.pixel_attention = nn.Sequential(
            AttBlock(dim, PixelAttention),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, dim // 2)
        )

        if num_classes:
            self.fc = nn.Linear(dim, num_classes)

        self.criterion = nn.SmoothL1Loss()
        self.cos_sim = nn.CosineSimilarity()

    def forward(self, x, pixel_x, channel_x):
        x = self.backbone(x)
        basic_pixel_attn = self.pixel_attention(x)
        basic_channel_attn = self.channel_attention(x)

        pixel_attn = self.pixel_wise_forward(pixel_x)
        channel_attn = self.channel_wise_forward(channel_x)

        pixel_loss = self.criterion(pixel_attn, basic_pixel_attn)
        channel_loss = self.criterion(channel_attn, basic_channel_attn)

        loss = (pixel_loss + channel_loss) / 2
        return loss, self.cos_sim(pixel_attn, basic_pixel_attn), self.cos_sim(channel_attn, basic_channel_attn)

    def channel_wise_forward(self, x):
        x = self.backbone(x)
        channel = self.channel_attention(x)
        return channel

    def pixel_wise_forward(self, x):
        x = self.backbone(x)
        pixel = self.pixel_attention(x)
        return pixel

    def attention_forward(self, x):
        x = self.backbone(x)
        return self.pixel_attention(x), self.channel_attention(x)

    def regression(self, x):
        device = x.device
        x = self.backbone(x)
        x = torch.cat([self.pixel_attention(x), self.channel_attention(x)], dim=1).to(device)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = DualAttentionNet()
    out = model.regression(input)

    print(out[0].shape)
    print(out[1].shape)
