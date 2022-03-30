from torchvision import datasets, transforms
import os
import random
from torch.nn import init
from random import shuffle
import numpy as npz
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import config

params = config.parse_args()


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('LayerNorm') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def normalize_gradient(net_D, x, y, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    f = net_D(x, y, **kwargs)
    grad = torch.autograd.grad(
        f, [x, y], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.4):
        self.mode = mode
        self.root = root
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        if self.mode == 'train':
            self.GT_paths_test = list(map(lambda k: os.path.join(self.GT_paths, k), os.listdir(self.GT_paths)))
        else:
            self.GT_paths_test = list(map(lambda k: os.path.join(self.GT_paths, k), os.listdir(self.GT_paths)))
        self.image_size = image_size
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        filename = image_path.split('./dataset/train/')[-1][:-len(".png")]
        GT_path = self.GT_paths + filename + '.png'
        GT_path = self.GT_paths_test[index]
        image = Image.open(image_path)
        GT = Image.open(GT_path)
        if params.RGB is True:
            image = image.convert('RGB')
        # aspect_ratio = image.size[1] / image.size[0]
        Transform = []
        # ResizeRange = random.randint(300,320)
        # Transform.append(T.Resize((int(self.image_size * aspect_ratio), self.image_size)))
        p_transform = random.random()
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # Transform.append(T.RandomCrop(self.image_size))
            # Transform = T.Compose(Transform)
            # image = Transform(image)
            # GT = Transform(GT)
            rotate = random.random()
            if rotate < 0.25:
                image = image.transpose(Image.ROTATE_90)
                GT = GT.transpose(Image.ROTATE_90)
            elif 0.5 > rotate > 0.25:
                image = image.transpose(Image.ROTATE_180)
                GT = GT.transpose(Image.ROTATE_180)
            elif 0.75 > rotate > 0.5:
                image = image.transpose(Image.ROTATE_270)
                GT = GT.transpose(Image.ROTATE_270)
            # Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.0)
            # GT = Transform(GT)
            flip_left_right = random.random()
            if flip_left_right < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                GT = GT.transpose(Image.FLIP_LEFT_RIGHT)
            flip_up_down = random.random()
            if flip_up_down < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                GT = GT.transpose(Image.FLIP_TOP_BOTTOM)
            Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        GT = Transform(GT)
        if params.RGB is True:
            Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            image = Norm_(image)
        return image, GT

    def __len__(self):
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, shuffle=True, num_workers=0, mode='train', augmentation_prob=0.4):
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=True, pin_memory=False)
    return data_loader
