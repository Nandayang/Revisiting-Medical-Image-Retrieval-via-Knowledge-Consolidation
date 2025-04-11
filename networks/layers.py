from torch import nn, einsum
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn


class CrossAttention2d(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention2d, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Conv2d(in_dim1, k_dim * num_heads, kernel_size=1, bias=False)
        self.proj_k2 = nn.Conv2d(in_dim2, k_dim * num_heads, kernel_size=1, bias=False)
        self.proj_v2 = nn.Conv2d(in_dim2, v_dim * num_heads, kernel_size=1, bias=False)
        self.proj_o = nn.Conv2d(v_dim * num_heads, in_dim1, kernel_size=1, bias=False)

    def forward(self, x1, x2, mask=None):
        b, c1, h1, w1 = x1.shape
        b, c2, h2, w2 = x2.shape

        # Ensure the spatial dimensions match
        if h1 != h2 or w1 != w2:
            raise ValueError("Spatial dimensions of x1 and x2 must match")

        # Project inputs to query, key, and value
        q1 = self.proj_q1(x1).view(b, self.num_heads, self.k_dim, h1 * w1).permute(0, 1, 3,
                                                                                   2)  # (b, num_heads, h1*w1, k_dim)
        k2 = self.proj_k2(x2).view(b, self.num_heads, self.k_dim, h2 * w2).permute(0, 1, 3,
                                                                                   2)  # (b, num_heads, h2*w2, k_dim)
        v2 = self.proj_v2(x2).view(b, self.num_heads, self.v_dim, h2 * w2).permute(0, 1, 3,
                                                                                   2)  # (b, num_heads, h2*w2, v_dim)

        # Calculate attention scores
        attn = torch.matmul(q1, k2.transpose(-1, -2)) / self.k_dim ** 0.5  # (b, num_heads, h1*w1, h2*w2)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn, v2)  # (b, num_heads, h1*w1, v_dim)

        # Reshape back to 2D
        output = output.permute(0, 1, 3, 2).contiguous().view(b, self.num_heads * self.v_dim, h1, w1)

        # Project output back to the original input dimension
        output = self.proj_o(output)

        return output

class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d

class AdP(nn.Module):
    """
    Adaptive Preprocessing Module.

    Applies batch normalization, a ReLU activation, and adaptive average pooling
    if the input spatial dimensions do not match the target size.
    """

    def __init__(self, in_channel, target_size):
        super(AdP, self).__init__()
        self.target_size = target_size
        self.norm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        if x.shape[2:] != self.target_size:
            x = self.gap(x)
        return x


class Decoder(nn.Module):
    """
    Decoder Module for OOD detection
    """

    def __init__(self, in_channels, middle_channels, out_channels, norm=nn.InstanceNorm2d, scale=2):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = norm(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x):
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.act1(c)
        c = self.conv2(c)
        c = self.norm2(c)
        c = self.act2(c)
        up = self.up(c)
        return up

