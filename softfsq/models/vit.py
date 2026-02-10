import math

import einops
import torch
import torch.nn as nn

from softfsq.quantization import Quantizer


class PositionalEmbedding1D(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, alpha: float = 1.0):
        # x: (B, N, D)
        B, N, D = x.shape

        position = torch.arange(N, dtype=torch.float32, device=x.device)[:, None]  # (N, 1)

        # Create dimension indices
        multiplicator = -torch.log(torch.tensor(10000.0)) / D
        d_series = torch.arange(0, D, 2, dtype=torch.float32, device=x.device)  # (D/2,)
        div_term = torch.exp(multiplicator * d_series)  # (D/2,)

        # Compute positional encodings
        pe = torch.zeros(N, D, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add to input
        return x + alpha * self.dropout(pe.unsqueeze(0))  # (B, N, D)


class UpscaleBlock(nn.Module):

    def __init__(self, d_model: int, d_feedforward: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            d_model, d_model, kernel_size=7, groups=d_model, padding=3, output_padding=1, stride=2
        )
        self.act1 = nn.GELU()
        self.linear1 = nn.Conv2d(d_model, d_feedforward, kernel_size=1)
        self.act2 = nn.GELU()
        self.linear2 = nn.Conv2d(d_feedforward, d_out, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        if d_out != d_model:
            self.upscale_proj = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(d_model, d_out, kernel_size=1, bias=False),
            )
        else:
            self.upscale_proj = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        z = self.conv1(x)
        z = self.act1(z)
        z = self.linear1(z)
        z = self.act2(z)
        z = self.linear2(z)
        return self.upscale_proj(x) + self.dropout(z)


class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        quantizer: Quantizer,
        d_input: int,
        d_output: int,
        d_model: int,
        d_feedforward: int,
        nhead: int,
        num_layers: int,
        patch_size: int,
        dropout: float = 0.1,
        num_extra_tokens: int = 0,
        activation: str = 'gelu',
    ):
        super().__init__()

        if not num_layers % 2 == 0:
            raise ValueError('num_layers must be even, as it is split between encoder and decoder')

        self.quantizer = quantizer
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.dropout = dropout
        self.num_extra_tokens = num_extra_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers // 2,
        )
        self.dec_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers // 2,
        )
        self.pos_embedding = PositionalEmbedding1D(d_model, dropout)

        self.encoder = nn.Linear(patch_size * patch_size * d_input, d_model, bias=False)

        decoder_layers = []
        res = int(math.log2(patch_size))
        for i in range(res):
            decoder_layers.append(UpscaleBlock(d_model, d_model, d_model, dropout))
        decoder_layers += [nn.Conv2d(d_model, d_output, kernel_size=3, padding=1)]
        self.decoder = nn.Sequential(*decoder_layers)

        self.pre_quantization_proj = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Linear(d_feedforward, quantizer.codebook_dim),
        )
        self.post_quantization_proj = nn.Linear(quantizer.codebook_dim, d_model, bias=True)

        if num_extra_tokens > 0:
            self.extra_tokens = nn.Embedding(num_extra_tokens, d_model)
        else:
            self.extra_tokens = None

    def forward(self, x, quantize=True):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p1, p2 = self.patch_size, self.patch_size
        if H % p1 != 0 or W % p2 != 0:
            raise ValueError(f'Image dimensions ({H}x{W}) must be divisible by patch size ({p1}x{p2})')

        # Tokenize
        inp_tokens = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p1, p2=p2)
        inp_tokens = self.encoder(inp_tokens)  # (B, N, D)

        if self.num_extra_tokens:
            extra_tokens = self.extra_tokens.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_extra_tokens, D)
            tokens = torch.cat([inp_tokens, extra_tokens], dim=1)  # (B, N + num_extra_tokens, D)
        else:
            tokens = inp_tokens

        # Encode
        tokens = self.pos_embedding(tokens)  # (B, N, D)
        encoded = self.enc_transformer(tokens)

        # Quantize
        z = self.pre_quantization_proj(encoded)  # (B, N, codebook_dim)
        quant_res = self.quantizer(z)
        z_q = quant_res.values if quantize else quant_res.pre_quantization
        z_q = self.post_quantization_proj(z_q)  # (B, N, D)

        # Decode
        z_q = self.pos_embedding(z_q)  # re-add pos. embedding after quantization
        decoded = self.dec_transformer(z_q)  # (B, N, D)
        if self.num_extra_tokens:
            decoded = decoded[:, : inp_tokens.shape[1]]  # (B, N, D)

        # Detokenize
        out = einops.rearrange(decoded, 'b (h w) c -> b c h w', h=H // p1, w=W // p2)
        out = self.decoder(out)  # (B, C_out, H, W)

        return out, quant_res


def vit_S(
    input_dim: int,
    output_dim: int,
    quantizer: Quantizer,
    patch_size: int,
    dropout: float = 0.0,
    num_extra_tokens: int = 0,
):
    # ~61M parameters
    return ViTAutoencoder(
        quantizer=quantizer,
        d_input=input_dim,
        d_output=output_dim,
        d_model=768,
        d_feedforward=2048,
        nhead=12,
        num_layers=10,
        patch_size=patch_size,
        dropout=dropout,
        num_extra_tokens=num_extra_tokens,
    )


def vit_B(
    input_dim: int,
    output_dim: int,
    quantizer: Quantizer,
    patch_size: int,
    dropout: float = 0.0,
    num_extra_tokens: int = 0,
):
    # ~90.5M parameters
    return ViTAutoencoder(
        quantizer=quantizer,
        d_input=input_dim,
        d_output=output_dim,
        d_model=768,
        d_feedforward=3072,
        nhead=12,
        num_layers=12,
        patch_size=patch_size,
        dropout=dropout,
        num_extra_tokens=num_extra_tokens,
    )
