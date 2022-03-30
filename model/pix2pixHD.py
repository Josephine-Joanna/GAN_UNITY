from functools import partial
import torch.nn as nn


# ------------------------------------------------------------------------------
# [2] Set the Normalization method for the input layer

def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


# ------------------------------------------------------------------------------
# [3] Set the Padding method for the input layer

def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


class Generator_HD(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator_HD, self).__init__()

        act = nn.ReLU(inplace=True)
        input_ch = input_dim
        n_gf = num_filter
        norm = get_norm_layer('InstanceNorm2d')
        output_ch = output_dim
        pad = get_pad_layer('reflection')

        model = []
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(4):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(9):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for _ in range(4):
            model += [nn.ConvTranspose2d(n_gf, n_gf // 2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf // 2), act]
            n_gf //= 2

        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
