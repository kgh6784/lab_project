from glob import glob

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10


class MyCIFAR10(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            basic_transform=None,
            pixel_transform=None,
            channel_transform=None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train, None, None, download)
        self.basic_transform = basic_transform
        self.pixel_transform = pixel_transform
        self.channel_transform = channel_transform

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.pixel_transform is None or self.channel_transform is None:
            return self.basic_transform(img)

        return self.basic_transform(img), self.pixel_transform(img), self.channel_transform(img)


def get_transform(size=96, s=1):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    resize = round(size * 1.13)

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    basic_aug = transforms.Compose([transforms.Resize((resize, resize)),
                                    transforms.CenterCrop((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    ])

    pos_aug = transforms.Compose([transforms.Resize((resize, resize)),
                                  transforms.RandomCrop((size, size), padding=4),
                                  transforms.RandomRotation(25),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean, std=std),
                                  ])

    channel_aug = transforms.Compose([transforms.Resize((resize, resize)),
                                      transforms.CenterCrop((size, size)),
                                      transforms.RandomApply([color_jitter], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std),
                                      ])
    return basic_aug, pos_aug, channel_aug


def make_images():
    imgs = glob('sample_images/*')
    print(imgs)
    basic_aug, pos_aug, channel_aug = get_transform(112)
    for i, img in enumerate(imgs):
        img = Image.open(img)
        img1 = basic_aug(img)
        img2 = pos_aug(img)
        img3 = channel_aug(img)
        print(i)
        img1.save(f'transforms_images/{i}.jpg')
        img2.save(f'transforms_images/pos_{i}.jpg')
        img3.save(f'transforms_images/cha_{i}.jpg')


if __name__ == '__main__':
    make_images()
