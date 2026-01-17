from functools import partial

from torch import nn


class DiscriminatorHingeLoss(nn.Module):
    """
    # TODO: Where is this from? WGAN?
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, logits_real, logits_fake):
        loss_real = self.relu(1 - logits_real).mean()
        loss_fake = self.relu(1 + logits_fake).mean()
        return 0.5 * (loss_real + loss_fake)


class DiscriminatorVanillaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, logits_real, logits_fake):
        loss_real = nn.functional.softplus(-logits_real).mean()
        loss_fake = nn.functional.softplus(logits_fake).mean()
        return 0.5 * (loss_real + loss_fake)


class GeneratorWGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_fake):
        return -logits_fake.mean()


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_dim=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_dim (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1

        self.in_transform = nn.Sequential(
            nn.Conv2d(input_dim, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        )

        layers = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        self.main = nn.Sequential(*layers)

        self.out_transform = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        )

        self.weights_init()

    def weights_init(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.in_transform(x)  # B,C,H,W
        x = self.main(x)  # B,C,H,W
        x = self.out_transform(x)  # B,1,H,W
        return x


class FasterNLayerDiscriminator(nn.Module):
    """
    ndf: 64 -> 32
    n_layers: 3 -> 4
    BatchNorm -> GroupNorm
    kernel: 4 -> 3
    receptive field: 70x70 -> 63x63
    params: 2.8M -> 3.9M/1M
    """

    def __init__(self, input_dim=3, start_channels=64, n_layers=4):
        super().__init__()

        norm_layer = partial(nn.GroupNorm, 32)
        use_bias = False

        num_channels = start_channels

        self.in_transform = nn.Sequential(
            nn.Conv2d(input_dim, num_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        layers = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            num_channels *= 2
            layer = nn.Sequential(
                nn.Conv2d(num_channels // 2, num_channels, 3, stride=2, padding=1, bias=use_bias),
                norm_layer(num_channels),
                nn.LeakyReLU(0.2, True),
            )
            layers += [layer]

        self.main = nn.Sequential(*layers)

        self.out_transform = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(num_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_channels, 1, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif classname.find('GroupNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.in_transform(x)  # B,C,H,W
        x = self.main(x)  # B,C,H,W
        x = self.out_transform(x)  # B,1,H,W
        return x
