import torch
import torch.nn as nn
import einops

import dmlcloud as dml

from softfsq.quantization import Quantizer


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=64, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scaling = nn.Parameter(0.2 * torch.ones(in_channels))
        self.conv = torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x_hr = nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear')
        return self.scaling[None, :, None, None] * self.conv(x) + x_hr


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scaling = nn.Parameter(0.2 * torch.ones(in_channels))
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x_lr = x[:,:,::2,::2]
        return self.scaling[None, :, None, None] * self.conv(x) + x_lr


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.activation = torch.nn.SiLU()
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.scaling = nn.Parameter(0.2 * torch.ones(out_channels))
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + self.scaling[None, :, None, None] * h



class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        in_channels,
        resolution,
        z_channels,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.conv_out = torch.nn.Sequential(
            Normalize(block_in),
            torch.nn.SiLU(),
            torch.nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        in_channels,
        resolution,
        z_channels,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.conv_out = torch.nn.Sequential(
            Normalize(block_in),
            torch.nn.Conv2d(block_in, block_in, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.SiLU(),
            torch.nn.Conv2d(block_in, out_ch, kernel_size=1),
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        with torch.amp.autocast(h.device.type, enabled=False):
            h = 0.25 * self.conv_out(h.float())
        return h


class TamingVQGAN(nn.Module):
    def __init__(self, ddconfig, quantizer: Quantizer):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantizer = quantizer
        self.post_quant_conv = nn.Sequential(
            nn.Conv2d(quantizer.codebook_dim, ddconfig['z_channels'], kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.register_buffer('step', torch.tensor(0, dtype=torch.long), persistent=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def last_layer_weights(self):
        return self.decoder.conv_out[-1].weight

    def forward(self, x, quantize=True):
        if self.training:
            self.step += 1

        z = self.encoder(x)

        z = einops.rearrange(z, 'B C H W -> B H W C').contiguous()
        quant_res = self.quantizer(z)
        x = quant_res.values if quantize else quant_res.pre_quantization
        x = einops.rearrange(x, 'B H W C -> B C H W').contiguous()

        x = self.post_quant_conv(x)
        decoded = self.decoder(x)
        return decoded, quant_res


def taming_f16(input_dim: int, output_dim: int, quantizer: Quantizer):
    ddconfig = {
        'z_channels': quantizer.codebook_dim,
        'resolution': 256,
        'in_channels': input_dim,
        'out_ch': output_dim,
        'ch': 128,
        'ch_mult': [1, 1, 2, 2, 4],
        'num_res_blocks': 2,
        'dropout': 0.0,
    }
    return TamingVQGAN(ddconfig, quantizer)


def taming_f8(input_dim: int, output_dim: int, quantizer: Quantizer):
    ddconfig = {
        'z_channels': quantizer.codebook_dim,
        'resolution': 256,
        'in_channels': input_dim,
        'out_ch': output_dim,
        'ch': 128,
        'ch_mult': [1, 1, 2, 4],
        'num_res_blocks': 2,
        'dropout': 0.0,
    }
    return TamingVQGAN(ddconfig, quantizer)


def taming_f4(input_dim: int, output_dim: int, quantizer: Quantizer):
    ddconfig = {
        'z_channels': quantizer.codebook_dim,
        'resolution': 256,
        'in_channels': input_dim,
        'out_ch': output_dim,
        'ch': 128,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'dropout': 0.0,
    }
    return TamingVQGAN(ddconfig, quantizer)


def taming_f4_L(input_dim: int, output_dim: int, quantizer: Quantizer):
    ddconfig = {
        'z_channels': quantizer.codebook_dim,
        'resolution': 256,
        'in_channels': input_dim,
        'out_ch': output_dim,
        'ch': 160,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 3,
        'dropout': 0.0,
    }
    return TamingVQGAN(ddconfig, quantizer)
