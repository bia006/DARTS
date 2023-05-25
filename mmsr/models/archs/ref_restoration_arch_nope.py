import mmsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from torch.nn import functional as F
import math
import einops
from typing import Tuple, Optional, List, Union, Any
from .CustomLayers import EqualLinear, PixelNorm
import torchvision.ops
from .op import fused_leaky_relu, upfirdn2d
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as init

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        # self.norm = nn.InstanceNorm1d(in_channel)
        self.norm = nn.LayerNorm(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).squeeze(-1)
        # print ('style', style.shape)
        gamma, beta = style.chunk(2, 2)
        # print ('gamma', gamma.shape, beta.shape)

        # out = self.norm(input).permute(0, 2, 1)
        out = self.norm(input)
        # print ('out', out.shape)
        out = gamma * out + beta
        # print ('last_out', out.shape)

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.is_upsample = upsample
        self.resolution = resolution

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = nn.Conv2d(in_channel, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)

            out = out + skip
        return out
    
    def flops(self):
        m = self.conv
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1
        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        flops = 1 * self.resolution * self.resolution * 3 * (m.in_channels // m.groups * kernel_ops + bias_ops)
        if self.is_upsample:
            # there is a conv used in upsample
            w_shape = (1, 1, 4, 4)
            kernel_ops = torch.zeros(w_shape[2:]).numel()  # Kw x Kh
            # N x Cout x H x W x  (Cin x Kw x Kh + bias)
            flops = 1 * 3 * (2 * self.resolution + 3) * (2 *self.resolution + 3) * (3 * kernel_ops)
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print ('B,H,W,C', x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size)) # for plain swin TR
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def unfold(input: torch.Tensor,
           window_size: int,
           kernel_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    channels, height, width = input.shape  # type: int, int, int
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=0, size=window_size, step=window_size) \
        .unfold(dimension=0, size=kernel_size, step=kernel_size)

    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(5, 0, 1, 4, 2, 3)
    B, H, W, H_, W_, C = output.shape
    output = output.reshape(-1, H * W, H_ * W_, C)

    return output


def fold(input: torch.Tensor,
         window_size: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    # H, W: int = int(input.shape[2] // (window_size), int(input.shape[3]// window_size)
    # Reshape input to
    output: torch.Tensor = input.view(-1, input.shape[2] // window_size, window_size, input.shape[3] // window_size, window_size, channels).contiguous()
    B, H, W, H_, W_, C = output.shape
    output: torch.Tensor = output.reshape(B*W*W_, H*H_, channels)
    return output


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., deformable_groups=1):

        super().__init__()
        ### Standard attn block 
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.n_head_channels = self.head_dim
        self.nc = self.n_head_channels * self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.num_heads // self.n_groups
        self.stride = stride
        self.offset_range_factor = offset_range_factor
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe
        self.deformable_groups = deformable_groups

        ksizes = [9, 7, 5, 3, 3]
        kk = ksizes[stage_idx]

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.q = self.qkv[0]
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)    

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = DeformableConv2d(
            self.nc, self.nc
        )

        self.proj_k = DeformableConv2d(
            self.nc, self.nc
        )

        self.proj_v = DeformableConv2d(
            self.nc, self.nc
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.conv_offset_mask = nn.Conv2d(
            self.nc//2,
            self.nc,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.init_offset()

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.num_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
    
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref  

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        # x1: torch.Tensor = q.unfold(dimension=1, size=self.window_size[0], step=self.window_size[1])
        x1: torch.Tensor = q.unfold(dimension=1, size=self.window_size[0], step=self.window_size[1])
        b, H, c, W = x1.shape
        x1 = x1.reshape(b, c, H, W)
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        # B_, N, C = q.shape

        # ####### Begining of Masi's attn block
        # q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # # print ('qqq', q.shape)
        # k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn_bias = relative_position_bias

        # attn = attn + attn_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.attn_drop(attn.softmax(dim=3))
        # else:
        #     attn = self.attn_drop(attn.softmax(dim=3))

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # x = self.proj_drop(self.proj(x)) # B' x N x C

        # return x, x, x
    
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        return x, x, x
        
        ######## end of Masi's attn block
        # x_standard = einops.rearrange(q, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W

        # ### deformable attn block 
        # x = x_standard

        # B, C, H, W = x.size()
        # dtype, device = x.dtype, x.device
        
        # q = self.proj_q(x)
        # q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        # offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        # Hk, Wk = offset.size(2), offset.size(3)
        # n_sample = Hk * Wk
        
        # if self.offset_range_factor > 0:
        #     offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
        #     offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        # offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        # if self.no_off:
        #     offset = offset.fill(0.0)
            
        # if self.offset_range_factor >= 0:
        #     pos = offset + reference
        # else:
        #     pos = (offset + reference).tanh()
        
        # x_sampled = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
        #     grid=pos[..., (1, 0)], # y, x -> x, y
        #     mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        # x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # q1 = q.reshape(B * self.num_heads, self.n_head_channels, H * W)
        # k1 = self.proj_k(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        # v1 = self.proj_v(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)


        # x_standard = einops.rearrange(k, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W

        # ### deformable attn block 
        # x = x_standard

        # B, C, H, W = x.size()
        # dtype, device = x.dtype, x.device
        
        # q = self.proj_q(x)
        # q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        # offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        # Hk, Wk = offset.size(2), offset.size(3)
        # n_sample = Hk * Wk
        
        # if self.offset_range_factor > 0:
        #     offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
        #     offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        # offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        # if self.no_off:
        #     offset = offset.fill(0.0)
            
        # if self.offset_range_factor >= 0:
        #     pos = offset + reference
        # else:
        #     pos = (offset + reference).tanh()
        
        # x_sampled = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
        #     grid=pos[..., (1, 0)], # y, x -> x, y
        #     mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        # x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # q2 = q.reshape(B * self.num_heads, self.n_head_channels, H * W)
        # k2 = self.proj_k(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        # v2 = self.proj_v(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        
        # attn = torch.einsum('b c m, b c n -> b m n', q1, k2) # B * h, HW, Ns
        # attn = attn.mul(self.scale)
        
        # if self.use_pe:
            
        #     if self.dwc_pe:
        #         residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.num_heads, self.n_head_channels, H * W)
        #     elif self.fixed_pe:
        #         rpe_table = self.rpe_table
        #         attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        #         attn = attn + attn_bias.reshape(B * self.num_heads, H * W, self.n_sample)
        #     else:
        #         rpe_table = self.rpe_table
        #         rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
        #         q_grid = self._get_ref_points(H, W, B, dtype, device)
                
        #         displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
        #         attn_bias = F.grid_sample(
        #             input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
        #             grid=displacement[..., (1, 0)],
        #             mode='bilinear', align_corners=True
        #         ) # B * g, h_g, HW, Ns
                
        #         attn_bias = attn_bias.reshape(B * self.num_heads, H * W, n_sample)
                
        #         attn = attn + attn_bias

        # attn = F.softmax(attn, dim=2)
        # attn = self.attn_drop(attn)
        
        # out = torch.einsum('b m n, b c n -> b c m', attn, v2)
        
        # if self.use_pe and self.dwc_pe:
        #     out = out + residual_lepe
        # out = out.reshape(B, C, H, W)
        
        # y = self.proj_drop(self.proj_out(out))
        # B, C, H, W = y.shape
        # y = y.reshape(-1, H * W, C)
          
        # return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RefSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, window_size, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm2 = AdaptiveInstanceNorm(self.dim, self.dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_groups=n_groups, stage_idx=stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.linear = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = norm_layer(dim)
        # self.norm1 = nn.InstanceNorm1d(dim)

        # self.norm2 = norm_layer(dim)
        # self.norm2 = nn.GELU()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask1 = None
        attn_mask2 = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                            self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.fused_window_process = fused_window_process

    def forward(self, x, ref):
        H, W = self.input_resolution
       
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        x = self.norm1(x)
        ref = self.norm1(ref)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

        qkv_ref = self.qkv(ref).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_ref1 = qkv_ref[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_ref2 = torch.roll(qkv_ref[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_ref2 = qkv_ref[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_ref2)

        x1, _, _ = self.attn(q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2, _, _ = self.attn(q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C//2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C//2), self.window_size, H, W)

        # cyclic shift
        if self.shift_size > 0:
            #x1 = torch.roll(x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            #x1 = x1
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C//2), x2.reshape(B, H * W, C//2)], dim=2)
        x = self.proj(x)
        

        # q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_ref1)
        # q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_ref2)

        # x3, _, _ = self.attn(q1_windows, k1_windows, v1_windows, self.attn_mask1)
        # x4, _, _ = self.attn(q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        # x3 = window_reverse(x3.view(-1, self.window_size * self.window_size, C//2), self.window_size, H, W)
        # x4 = window_reverse(x4.view(-1, self.window_size * self.window_size, C//2), self.window_size, H, W)

        # # cyclic shift
        # if self.shift_size > 0:
        #     #x1 = torch.roll(x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        #     x4 = torch.roll(x4, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     #x1 = x1
        #     x4 = x4
                
        # x = torch.cat([x3.reshape(B, H * W, C // 2), x4.reshape(B, H * W, C // 2)], dim=2)
        # x_ = self.proj(x)

        # x = torch.cat((x, x_), 2) 
        # x = self.linear(x)

        # x = x.view(B, H * W, C)
        # x = shortcut + self.drop_path(x)

        # # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # x = shortcut + x + self.drop_path(x)
        x = shortcut + x
        # ref = shortcut_ref + ref + self.drop_path(ref)

        # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # ref = ref + self.drop_path(self.mlp(self.norm2(ref)))
        x = x + self.drop_path(self.mlp(self.norm1(x)))

        return x, ref

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe,                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RefSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 n_groups=n_groups, stage_idx= stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe, 
                                 shift_size = window_size//2,
                                #  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=dim)
        else:
            self.upsample = None
        if all(v is not None for v in [downsample, upsample]):
            self.linear = SinusoidalPositionalEmbedding(embedding_dim=dim//2, padding_idx=0, init_size=dim // 2)
        

    def forward(self, x, ref):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, ref = blk(x, ref)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x, ref

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 160.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=160, patch_size=4, in_chans=3, embed_dim=512, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # print ('111', x.shape)
        # B, H, W, C = x.shape
        # x = x.view(B, C, H, W)
        # print ('XXX', x.shape)
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
        

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 160.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=160, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) #(160, 160)
        patch_size = to_2tuple(patch_size) # (4, 4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] #(56, 56)
        self.img_size = img_size 
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] #(56^2)

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.batch_dim = [int(x / self.patch_size[0]) for x in self.img_size]

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        # x = x.transpose(-1, -2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, blur_kernel=[1, 3, 3, 1], out_dim=None, scale_factor=2):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsample = nn.PixelShuffle(2)
        # self.conv = nn.Conv2d(dim//4, dim, 1, 1, 0)
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE    
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # LN
        flops = 4 * H * W * self.dim
        # proj
        flops += 4 * H * W * self.dim * (self.out_dim)
        # SPE
        flops += 4 * H * W * 2
        # bilinear
        flops += 4 * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops


class ConstantInput(nn.Module):
    def __init__(self, channel, size):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.norm = nn.InstanceNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class Up_Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

def default_init_weights(module_list, scale=1):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(basic_block, n_basic_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        n_basic_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(n_basic_blocks):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat
    
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        sn (bool): Whether to use spectral norm. Default: False.
        n_power_iterations (int): Used in spectral norm. Default: 1.
        sn_bias (bool): Whether to apply spectral norm to bias. Default: True.
    """

    def __init__(self,
                 nf=64,
                 res_scale=1,
                 pytorch_init=False,
                 sn=False,
                 n_power_iterations=1,
                 sn_bias=True):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sn:
            self.conv1 = spectral_norm(
                self.conv1,
                name='weight',
                n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(
                self.conv2,
                name='weight',
                n_power_iterations=n_power_iterations)
            if sn_bias:
                self.conv1 = spectral_norm(
                    self.conv1,
                    name='bias',
                    n_power_iterations=n_power_iterations)
                self.conv2 = spectral_norm(
                    self.conv2,
                    name='bias',
                    n_power_iterations=n_power_iterations)
        self.relu = nn.ReLU(inplace=True)

        if not sn and not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 160
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=40, patch_size=4, in_chans=6, in_channel=3, num_classes=65536, embed_dim=48, depths=[2, 2, 2, 2], num_heads=[24, 16, 24, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, upsample=False, patch_norm=True,
                 use_checkpoint=False, use_deformable_block: bool = True, fused_window_process=False, stage_idx=[0, 1, 2, 3], strides=[1, 1, 1, 1], groups=[6, 6, 6, 6], use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False], fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 offset_range_factor=[1, 2, 3, 4], 
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.patch_embed_ref = PatchEmbed(
            img_size=img_size*2, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches_ref = self.patch_embed_ref.num_patches
        patches_resolution_ref = self.patch_embed_ref.patches_resolution
        self.patches_resolution_ref = patches_resolution_ref

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_ref = nn.Parameter(torch.zeros(1, num_patches_ref, embed_dim))
            trunc_normal_(self.absolute_pos_embed_ref, std=.02)

        self.patch_embed_ref_g = PatchEmbed(
            img_size=img_size*4, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches_ref_g = self.patch_embed_ref_g.num_patches
        patches_resolution_ref_g = self.patch_embed_ref_g.patches_resolution
        self.patches_resolution_ref_g = patches_resolution_ref_g
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_ref_g = nn.Parameter(torch.zeros(1, num_patches_ref_g, embed_dim))
            trunc_normal_(self.absolute_pos_embed_ref_g, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # self.patch_dim = [int((x / patch_size)/2**3) for x in img_shape]
        # self.patch_dim = img_size // patch_size

        # build layers
        # in_channels = [
        #     384,
        #     96, 
        #     96, 
        #     96, 
        #     24
        #     ] 

        in_channels = [
            96,
            96, 
            96,  
            96
            ] 
        
        self.layers = nn.ModuleList()
        self.layers_ref = nn.ModuleList()
        self.layers_ref_g = nn.ModuleList()
                
        ### student
        for i_layer in range(self.num_layers - 2):
        # for i_layer in range(self.num_layers - 2):
            in_channel = in_channels[i_layer]
            layer = BasicLayer(dim=in_channel,
                                # input_resolution=(patches_resolution_ref[0] * (2 ** i_layer),
                                #                   patches_resolution_ref[1] * (2 ** i_layer)),
                                input_resolution=(img_size*2* (2 ** i_layer), img_size*2* (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                stride=strides[i_layer],
                                n_groups=groups[i_layer],
                                offset_range_factor=offset_range_factor[i_layer],
                                use_pe=use_pes[i_layer],
                                stage_idx=stage_idx[i_layer],
                                fixed_pe=fixed_pes[i_layer],
                                no_off=no_offs[i_layer],
                                dwc_pe=dwc_pes[i_layer],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                upsample=BilinearUpsample if (i_layer < self.num_layers - 3) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process)
            self.layers.append(layer)

        for i_layer_ref in range(self.num_layers - 2):
            in_channel = in_channels[i_layer_ref]
            layer_ref = BasicLayer(dim=in_channel,
                                        # input_resolution=(patches_resolution[0] * (2 ** i_layer_ref),
                                        #                 patches_resolution[1] * (2 ** i_layer_ref)),
                                        input_resolution=(img_size* (2 ** i_layer_ref), img_size* (2 ** i_layer_ref)),
                                        # input_resolution=(40, 40),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size,
                                        stride=strides[i_layer_ref],
                                        n_groups=groups[i_layer_ref],
                                        offset_range_factor=offset_range_factor[i_layer_ref],
                                        use_pe=use_pes[i_layer_ref],
                                        stage_idx=stage_idx[i_layer_ref],
                                        fixed_pe=fixed_pes[i_layer_ref],
                                        no_off=no_offs[i_layer_ref],
                                        dwc_pe=dwc_pes[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        upsample=BilinearUpsample if (i_layer < self.num_layers - 2) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)
            self.layers_ref.append(layer_ref)

        for i_layer_ref in range(self.num_layers - 3):
            in_channel = in_channels[i_layer_ref]
            layer_ref_g = BasicLayer(dim=in_channel,
                                        # input_resolution=(patches_resolution_ref_g[0] * (2 ** i_layer_ref),
                                        #                 patches_resolution_ref_g[1] * (2 ** i_layer_ref)),
                                        input_resolution=(img_size*4*(2 ** i_layer_ref), img_size*4*(2 ** i_layer_ref)),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size,
                                        stride=strides[i_layer_ref],
                                        n_groups=groups[i_layer_ref],
                                        offset_range_factor=offset_range_factor[i_layer_ref],
                                        use_pe=use_pes[i_layer_ref],
                                        stage_idx=stage_idx[i_layer_ref],
                                        fixed_pe=fixed_pes[i_layer_ref],
                                        no_off=no_offs[i_layer_ref],
                                        dwc_pe=dwc_pes[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        upsample=BilinearUpsample if (i_layer < self.num_layers - 4) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)
            self.layers_ref_g.append(layer_ref_g)
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=in_channels[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=1e-6) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(in_channels[-1]) # final norm layer
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=4, stride=4)
        self.proj_1 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1, stride=1)
        self.proj_2 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels[-1], 4 * in_channels[-1], 3, 1, 1)
        self.upsample = nn.PixelShuffle(2)
        
        self.conv1 = nn.Conv2d(in_channels[-1]*2, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels[-1], in_channels[-1], 3, 1, 1)
        self.conv = nn.Conv2d(in_channels[-1]//4, in_channels[-1], 3, 1, 1)
        self.conv_31 = nn.Conv2d(256, in_channels[-1], 3, 1, 1)
        self.conv_21 = nn.Conv2d(128, in_channels[-1], 3, 1, 1)
        self.conv_11 = nn.Conv2d(64, in_channels[-1], 3, 1, 1)
        # self.norm_last = norm_layer(self.num_features//2**5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels[-1], num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Linear(in_channels[-1], num_classes) 
        self.head_in = nn.Conv2d(3, in_channels[-1], 3, 1, 1) 
        self.head_1 = nn.Conv2d(352, in_channels[-1], 3, 1, 1)
        self.head_2 = nn.Conv2d(224, in_channels[-1], 3, 1, 1)
        self.head_3 = nn.Conv2d(131, in_channels[-1], 3, 1, 1)
        self.head_4 = nn.Conv2d(160, in_channels[-1], 1, 1, 0)
        self.out = nn.Conv2d(in_channels[0], 3, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=96, n_blocks=16)

        self.small_offset_conv1 = nn.Conv2d(
            96 + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 96, 3, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, lr, ref1, ref, ref_g, img_ref_feat, truncation=1, truncation_latent=None): 
        student_output = []    
        lr_output = []
        ref_output = []
        shortcut = lr
      
        lr = self.content_extractor(lr)
        head_lr = lr
        
        ref1 = self.conv_31(img_ref_feat['relu3_1'])
        ref = self.conv_21(img_ref_feat['relu2_1'])
        ref_g = self.conv_11(img_ref_feat['relu1_1'])
        
        # lr = self.patch_embed(lr)
        # if self.ape:
        #     lr = lr + self.absolute_pos_embed
        # lr = self.pos_drop(lr)
        # ref1 = self.patch_embed(ref1)
        # if self.ape:
        #     ref1 = ref1 + self.absolute_pos_embed
        # ref1 = self.pos_drop(ref)
        base = F.interpolate(shortcut, None, 4, 'bilinear', False)

        B, C, H, W = head_lr.shape
        lr = head_lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref1 = ref1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
               
        for layer in self.layers_ref:
            lr, ref1 = layer(lr, ref1)  
        for layer in self.layers_ref_g:   
            lr, ref1 = layer(lr, ref1) 
        lr_1 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous()
        # lr = self.conv1(torch.cat((lr, head_lr), 1)) 
        lr_up1 = self.upsample(self.conv4(head_lr))

        B, C, H, W = lr_up1.shape
        lr = lr_up1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref = ref.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        for layer in self.layers:
            lr, ref = layer(lr, ref)    
        lr_2 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous()
        # lr = self.conv1(torch.cat((lr, lr_up), 1))
        lr_up2 = self.upsample(self.conv4(lr_up1))
        
        B, C, H, W = lr_up2.shape
        lr = lr_up2.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_g = ref_g.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)     
       
        for layer in self.layers_ref_g:
            lr, ref_g = layer(lr, ref_g) 
    
        lr_3 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous() 
        lr_distill = lr_1 + lr_2 + lr_3
        lr = self.out(lr_1 + lr_2 + lr_3) + base
       
        B, C, H, W = lr_distill.shape
        x_lr = lr_distill.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        x = self.norm(x_lr)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        lr_output.append(x)

        return lr, lr_output

    def forward(self, lr, ref1, ref, ref_g, img_ref_feat, truncation=1, truncation_latent=None):
        out, lr_output = self.forward_features(lr, ref1, ref, ref_g, img_ref_feat)

        return out, lr_output             

    def flops(self):
        flops = 0
        flops += self.patch_embed_s.flops() 
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution_s[0] * self.patches_resolution_s[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

def right_rotate(lists, n): 
        output_list = [] 
        x= len(lists)
        
        for item in range(x - n, x): 
            output_list.append(lists[item]) 
        
        
        for item in range(0, x - n):  
            output_list.append(lists[item]) 
            
        return output_list 

class SwinTransformer_T(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 160
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=40, patch_size=4, in_chans=6, in_channel=3, num_classes=65536, embed_dim=48, depths=[2, 2, 2, 2], num_heads=[16, 16, 12, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, upsample=False, patch_norm=True,
                 use_checkpoint=False, use_deformable_block: bool = True, fused_window_process=False, stage_idx=[0, 1, 2, 3], strides=[1, 1, 1, 1], groups=[6, 6, 6, 6], use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False], fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 offset_range_factor=[1, 2, 3, 4], 
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.patch_embed_ref = PatchEmbed(
            img_size=img_size*2, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches_ref = self.patch_embed_ref.num_patches
        patches_resolution_ref = self.patch_embed_ref.patches_resolution
        self.patches_resolution_ref = patches_resolution_ref

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_ref = nn.Parameter(torch.zeros(1, num_patches_ref, embed_dim))
            trunc_normal_(self.absolute_pos_embed_ref, std=.02)

        self.patch_embed_ref_g = PatchEmbed(
            img_size=img_size*4, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches_ref_g = self.patch_embed_ref_g.num_patches
        patches_resolution_ref_g = self.patch_embed_ref_g.patches_resolution
        self.patches_resolution_ref_g = patches_resolution_ref_g
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_ref_g = nn.Parameter(torch.zeros(1, num_patches_ref_g, embed_dim))
            trunc_normal_(self.absolute_pos_embed_ref_g, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # self.patch_dim = [int((x / patch_size)/2**3) for x in img_shape]
        # self.patch_dim = img_size // patch_size

        # build layers
        # in_channels = [
        #     384,
        #     96, 
        #     96, 
        #     96, 
        #     24
        #     ] 

        in_channels = [
            96,
            96, 
            96,  
            96
            ] 
        
        self.layers = nn.ModuleList()
        self.layers_ref = nn.ModuleList()
        self.layers_ref_g = nn.ModuleList()
                
        ### student
        for i_layer in range(self.num_layers - 2):
        # for i_layer in range(self.num_layers - 2):
            in_channel = in_channels[i_layer]
            layer = BasicLayer(dim=in_channel,
                                # input_resolution=(patches_resolution_ref[0] * (2 ** i_layer),
                                #                   patches_resolution_ref[1] * (2 ** i_layer)),
                                input_resolution=(img_size*2* (2 ** i_layer), img_size*2* (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                stride=strides[i_layer],
                                n_groups=groups[i_layer],
                                offset_range_factor=offset_range_factor[i_layer],
                                use_pe=use_pes[i_layer],
                                stage_idx=stage_idx[i_layer],
                                fixed_pe=fixed_pes[i_layer],
                                no_off=no_offs[i_layer],
                                dwc_pe=dwc_pes[i_layer],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                upsample=BilinearUpsample if (i_layer < self.num_layers - 3) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process)
            self.layers.append(layer)

        for i_layer_ref in range(self.num_layers - 2):
            in_channel = in_channels[i_layer_ref]
            layer_ref = BasicLayer(dim=in_channel,
                                        # input_resolution=(patches_resolution[0] * (2 ** i_layer_ref),
                                        #                 patches_resolution[1] * (2 ** i_layer_ref)),
                                        input_resolution=(img_size* (2 ** i_layer_ref), img_size* (2 ** i_layer_ref)),
                                        # input_resolution=(40* (2 ** i_layer_ref), 40* (2 ** i_layer_ref)) if (i_layer < self.num_layers - 3) else (160, 160),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size,
                                        stride=strides[i_layer_ref],
                                        n_groups=groups[i_layer_ref],
                                        offset_range_factor=offset_range_factor[i_layer_ref],
                                        use_pe=use_pes[i_layer_ref],
                                        stage_idx=stage_idx[i_layer_ref],
                                        fixed_pe=fixed_pes[i_layer_ref],
                                        no_off=no_offs[i_layer_ref],
                                        dwc_pe=dwc_pes[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        upsample=BilinearUpsample if (i_layer < self.num_layers - 2) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)
            self.layers_ref.append(layer_ref)

        for i_layer_ref in range(self.num_layers - 3):
            in_channel = in_channels[i_layer_ref]
            layer_ref_g = BasicLayer(dim=in_channel,
                                        # input_resolution=(patches_resolution_ref_g[0] * (2 ** i_layer_ref),
                                        #                 patches_resolution_ref_g[1] * (2 ** i_layer_ref)),
                                        input_resolution=(img_size*4*(2 ** i_layer_ref), img_size*4*(2 ** i_layer_ref)),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size,
                                        stride=strides[i_layer_ref],
                                        n_groups=groups[i_layer_ref],
                                        offset_range_factor=offset_range_factor[i_layer_ref],
                                        use_pe=use_pes[i_layer_ref],
                                        stage_idx=stage_idx[i_layer_ref],
                                        fixed_pe=fixed_pes[i_layer_ref],
                                        no_off=no_offs[i_layer_ref],
                                        dwc_pe=dwc_pes[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        upsample=BilinearUpsample if (i_layer < self.num_layers - 4) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)
            self.layers_ref_g.append(layer_ref_g)
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=in_channels[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=1e-6) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(in_channels[-1]) # final norm layer
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=4, stride=4)
        self.proj_1 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1, stride=1)
        self.proj_2 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels[-1], 4 * in_channels[-1], 1, 1, 0)
        self.upsample = nn.PixelShuffle(2)
        
        self.conv1 = nn.Conv2d(in_channels[-1]*2, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels[-1], in_channels[-1], 3, 1, 1)
        self.conv = nn.Conv2d(in_channels[-1]//4, in_channels[-1], 3, 1, 1)
        self.conv_31 = nn.Conv2d(256, in_channels[-1], 3, 1, 1)
        self.conv_21 = nn.Conv2d(128, in_channels[-1], 3, 1, 1)
        self.conv_11 = nn.Conv2d(64, in_channels[-1], 3, 1, 1)
        # self.norm_last = norm_layer(self.num_features//2**5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels[-1], num_classes) 
        self.head_in = nn.Conv2d(3, in_channels[-1], 1, 1, 0) 
        self.head_1 = nn.Conv2d(352, in_channels[-1], 1, 1, 0)
        self.head_2 = nn.Conv2d(224, in_channels[-1], 1, 1, 0)
        self.head_3 = nn.Conv2d(131, in_channels[-1], 1, 1, 0)
        self.head_4 = nn.Conv2d(160, in_channels[-1], 1, 1, 0)
        self.out = nn.Conv2d(in_channels[0], 3, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=96, n_blocks=8)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, lr, ref1, ref, ref_g, img_ref_feat, truncation=1, truncation_latent=None): 
        student_output = []    
        lr_output = []
        ref_output = []
        shortcut = lr
      
        lr = self.content_extractor(ref)
        head_lr = lr
        
        ref1 = self.conv_31(img_ref_feat['relu3_1'])
        # ref1 = self.content_extractor(ref1)
        # ref = self.content_extractor(ref)
        # ref_g = self.content_extractor(ref_g)
        ref = self.conv_21(img_ref_feat['relu2_1'])
        ref_g = self.conv_11(img_ref_feat['relu1_1'])
        
        # lr = self.patch_embed(lr)
        # if self.ape:
        #     lr = lr + self.absolute_pos_embed
        # lr = self.pos_drop(lr)
        # ref1 = self.patch_embed(ref1)
        # if self.ape:
        #     ref1 = ref1 + self.absolute_pos_embed
        # ref1 = self.pos_drop(ref)
        base = F.interpolate(shortcut, None, 4, 'bilinear', False)

        # B, C, H, W = lr.shape
        # lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        # ref1 = ref1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
       
        # for layer in self.layers_ref:
        #     lr, ref1 = layer(lr, ref1)  
        # for layer in self.layers_ref_g:   
        #     lr, ref1 = layer(lr, ref1) 
        # lr_1 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous()
        # # lr = self.conv1(torch.cat((lr, head_lr), 1)) 
        # lr_up1 = self.upsample(self.conv4(head_lr))

        # B, C, H, W = lr_up1.shape
        # lr = lr_up1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        # ref = ref.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        # for layer in self.layers:
        #     lr, ref = layer(lr, ref)    
        # lr_2 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous()
        # # lr = self.conv1(torch.cat((lr, lr_up), 1))
        # lr_up2 = self.upsample(self.conv4(lr_up1))
              
        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_g = ref_g.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)     
       
        for layer in self.layers_ref_g:
            lr, ref_g = layer(lr, ref_g) 
    
        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous() 
        # lr = lr_1 + lr_2 + lr_3

        B, C, H, W = lr.shape
        x_ref = lr.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        x = self.norm(x_ref)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        ref_output.append(x)

        return ref_output

    def forward(self, lr, ref1, ref, ref_g, img_ref_feat, truncation=1, truncation_latent=None):
        ref_output = self.forward_features(lr, ref1, ref, ref_g, img_ref_feat)

        return ref_output             

    def flops(self):
        flops = 0
        flops += self.patch_embed_s.flops() 
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution_s[0] * self.patches_resolution_s[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

"""
class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


class RestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_(
        )
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, img_ref_feat):
        
        
        # Args:
        #     x (Tensor): the input image of SRNTT.
        #     maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
        #         and relu1_1. depths of the maps are 256, 128 and 64
        #         respectively.
      

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset,
                                               img_ref_feat)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()

        # dynamic aggregation module for relu3_1 reference feature
        self.small_offset_conv1 = nn.Conv2d(
            ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(
            256,
            256,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for small scale restoration
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_small = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_offset_conv1 = nn.Conv2d(
            ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(
            128,
            128,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for medium scale restoration
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_medium = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(
            64,
            64,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for large scale
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_large = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset, img_ref_feat):
        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(
            self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],
                               pre_offset['relu3_1']))
        # small scale
        h = torch.cat([x, relu3_swapped_feat], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(
            self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                pre_offset['relu2_1']))
        # medium scale
        h = torch.cat([x, relu2_swapped_feat], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(
            self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                               pre_offset['relu1_1']))
        # large scale
        h = torch.cat([x, relu1_swapped_feat], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x
"""
