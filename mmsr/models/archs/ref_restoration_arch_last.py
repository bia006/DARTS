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
from .CustomLayers import EqualLinear
from .op import fused_leaky_relu, upfirdn2d

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
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).squeeze(-1)
        print ('style', style.shape)
        gamma, beta = style.chunk(2, 2)
        print ('gamma', gamma.shape, beta.shape)

        # out = self.norm(input).permute(0, 2, 1)
        out = self.norm(input)
        print ('out', out.shape)
        out = gamma * out + beta
        print ('last_out', out.shape)

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


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

    def __init__(self, dim, window_size, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

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

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
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

    def forward(self, q, k, v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        # z = int(math.log2(x.shape[2]//self.dim))
        # z_ = int(math.sqrt(x.shape[0]))
        x1: torch.Tensor = q.unfold(dimension=1, size=self.window_size[0], step=self.window_size[1])
        b, H, c, W = x1.shape
        x1 = x1.reshape(b, c, H, W)
        # print ('xxx', x.shape)
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        B_, N, C = q.shape
              
        # x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C
        # print ('total_1', x_total.shape)
        
        # x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')
        # print ('total_2', x_total.shape)
        # qkv_x = self.qkv(x_total)
        # print ('qkv_x', qkv_x.shape)

        ####### Begining of Masi's attn block
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print ('qqq', q.shape)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.num_heads) for t in [q, k, v]]

        # attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)
        # attn_ref = torch.einsum('b h m c, b h n c -> b h m n', q_ref, k_ref)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias

        attn = attn + attn_bias.unsqueeze(0)
        # attn_ref = attn_ref + attn_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.attn_drop(attn.softmax(dim=3))
        else:
            attn = self.attn_drop(attn.softmax(dim=3))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # attn = self.attn_drop(attn.softmax(dim=3))
        # attn_ref = self.attn_drop(attn_ref.softmax(dim=3))

        # x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        # x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj(x)) # B' x N x C
        x_standard = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W

        # print ('standard', x_standard.shape)

        #### deformable attn block 
        x = x_standard

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        if self.no_off:
            offset = offset.fill(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.num_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        if self.use_pe:
            
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.num_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.num_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.num_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        
        y = self.proj_drop(self.proj_out(out))
        B, C, H, W = y.shape
        y = y.reshape(-1, H * W, C)
          
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
        
        ######## end of Masi's attn block


        # B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)

        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # print ('111', x.shape)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # print ('222', x.shape)
        # # return x



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

        # self.norm1 = AdaptiveInstanceNorm(self.dim, self.dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_groups=n_groups, stage_idx=stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.linear = nn.Linear(dim * 2, dim)
        self.norm1 = norm_layer(dim)

        self.norm2 = norm_layer(dim)
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
        # print ('H, W', H, W)
        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        shortcut_ref = ref
        # x = self.norm1(x.transpose(-1, -2), ref).transpose(-1, -2).permute(0, 2, 1)
        x = self.norm1(x)
        ref = self.norm1(ref)

        qkv_x = self.qkv(x)
        qkv_x = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_x = qkv_x[:, :, :, :].reshape(3, B, H, W, C)

        qkv_ref = self.qkv(ref)
        qkv_ref = self.qkv(ref).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_ref = qkv_ref[:, :, :, :].reshape(3, B, H, W, C)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_x)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_ref)

        x1, _, _ = self.attn(q1_windows, k2_windows, v2_windows, self.attn_mask1)
        x2, _, _ = self.attn(q2_windows, k1_windows, v1_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)

        # cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = x1
            x2 = x2
                
        x = torch.cat([x1.reshape(B, H * W, C), ref.reshape(B, H * W, C)], dim=2)
        x = self.linear(x)

        ref = torch.cat([x2.reshape(B, H * W, C), ref.reshape(B, H * W, C)], dim=2)
        ref = self.linear(ref)

        x = shortcut + x + self.drop_path(x)
        ref = shortcut_ref + ref + self.drop_path(ref)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        ref = ref + self.drop_path(self.mlp(self.norm2(ref)))

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
    

    def forward(self, x, ref):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, ref = checkpoint.checkpoint(blk, x, ref)
            else:
                x, ref = blk(x, ref)
        if self.downsample is not None:
            x = self.downsample(x)
            ref = self.downsample(ref)
        if self.upsample is not None:
            x = self.upsample(x)
            ref = self.upsample(ref)
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
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        # self.upsample = Upsample(blur_kernel)
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

    def __init__(self, img_size=[56, 224], patch_size=[4, 1], in_chans=3, num_classes=65536, embed_dim=[384, 24], depths=[2, 2, 6, 2, 6], num_heads=[3, 6, 12, 24, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, upsample=False, patch_norm=True,
                 use_checkpoint=False, use_deformable_block: bool = True, fused_window_process=False, stage_idx=[0, 1, 2, 3, 4], strides=[1,1, 1, 1, 1], groups=[1, 1, 3, 6, 6], use_pes=[True, True, True, True, True], 
                 dwc_pes=[True, True, True, True, True], fixed_pes=[False, False, False, False, False],
                 no_offs=[False, False, False, False, False],
                 offset_range_factor=[1, 2, 3, 4, 5], 
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
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)


        self.patch_embed_ref = PatchEmbed(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=in_chans, embed_dim=embed_dim[1],
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches_ref = self.patch_embed_ref.num_patches
        patches_resolution_ref = self.patch_embed_ref.patches_resolution
        self.patches_resolution_ref = patches_resolution_ref
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_ref = nn.Parameter(torch.zeros(1, num_patches_ref, embed_dim[1]))
            trunc_normal_(self.absolute_pos_embed_ref, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # self.patch_dim = [int((x / patch_size)/2**3) for x in img_shape]
        # self.patch_dim = img_size // patch_size

        # build layers
        in_channels = [
            384,
            192, 
            96, 
            48, 
            24
            ] 
    
        up_inputs = [
            80, 
            160
            ]  
        up_dims = [
            96, 
            48
            ]  
        
        self.layers = nn.ModuleList()
                
        ### student
        for i_layer in range(self.num_layers):
            in_channel = in_channels[i_layer]
            layer = BasicLayer(dim=in_channel,
                                input_resolution=(patches_resolution[0] * (2 ** i_layer),
                                                  patches_resolution[1] * (2 ** i_layer)),
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
                                upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.layer_ref = BasicLayer(dim=in_channels[4],
                                        input_resolution=(patches_resolution_ref[0],
                                                        patches_resolution_ref[1]),
                                        depth=depths[4],
                                        num_heads=num_heads[4],
                                        window_size=window_size,
                                        stride=strides[4],
                                        n_groups=groups[4],
                                        offset_range_factor=offset_range_factor[2],
                                        use_pe=use_pes[4],
                                        stage_idx=stage_idx[4],
                                        fixed_pe=fixed_pes[4],
                                        no_off=no_offs[4],
                                        dwc_pe=dwc_pes[4],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:4]):sum(depths[:4 + 1])],
                                        norm_layer=norm_layer,
                                        # upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)

        self.norm1 = norm_layer(192)
        self.norm2 = norm_layer(96)
        self.norm3 = norm_layer(48)
        self.norm4 = norm_layer(24)
        self.out_norm = norm_layer(24)
        # self.norm_last = norm_layer(self.num_features//2**5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.first_layer = nn.Conv2d(3, 96, 3, 1, 1)
        self.head1 = nn.Linear(192, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(96, num_classes) if num_classes > 0 else nn.Identity()
        self.head3 = nn.Linear(48, num_classes) if num_classes > 0 else nn.Identity()
        self.head4 = nn.Linear(24, num_classes) if num_classes > 0 else nn.Identity()
        
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = nn.Conv2d(24, 3, 1, 1, 0)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward_features(self, lr, ref, ref_g): 
        student_output = []    
        shortcut = lr
        lr = self.patch_embed(lr)
        ref = self.patch_embed(ref)
        # print ('S111', lr.shape)
        ref_g = self.patch_embed_ref(ref_g)
        if self.ape:
            lr = lr + self.absolute_pos_embed
            ref = ref + self.absolute_pos_embed
            ref = ref + self.absolute_pos_embed_ref
            # print ('111', x.shape)
        lr = self.pos_drop(lr)
        ref = self.pos_drop(ref)
        ref_g = self.pos_drop(ref_g)

        for layer in self.layers:
            lr, ref = layer(lr, ref)
            # lr = lr * ref
            # new_lr = lr.permute(0,2,1)
            # if new_lr.shape[2] == 6400:
            #     new_lr = self.upsample_4(new_lr)
            # new_lr = new_lr.permute(0,2,1)

            # x = self.norm2(new_lr)  # B L C
            # x = self.avgpool(x.transpose(1, 2))  # B C 1
            # x = torch.flatten(x, 1)
            # x = self.head2(x)


            if lr.shape[2] == self.embed_dim[0] // 2:
                x = self.norm1(lr)  # B L C
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
                x = self.head1(x)
                student_output.append(x)

            elif lr.shape[2] == self.embed_dim[0] // 4:
                x = self.norm2(lr)  # B L C
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
                x = self.head2(x)
                student_output.append(x)

            elif lr.shape[2] == self.embed_dim[0] // 8:
                x = self.norm3(lr)  # B L C
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
                x = self.head3(x)
                student_output.append(x)

            elif lr.shape[2] == self.embed_dim[0] // 16:
                x = self.norm4(lr)  # B L C
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
                x = self.head4(x)
                student_output.append(x)

        # lr = self.out_norm(lr) 
        # ref_g, _ = self.layer_ref(ref_g, ref_g)
        # ref_g = self.out_norm(ref_g) 
        # lr = lr * lr

        lr = lr.transpose(-1, -2).contiguous().view(lr.shape[0], 24, 224, 224)
        # ref = ref.transpose(-1, -2).contiguous().view(ref.shape[0], 24, 224, 224)
        lr = self.out(lr)

        return lr, student_output

    def forward(self, lr, ref, ref_g):
        lr, student_output = self.forward_features(lr, ref, ref_g)

        return lr, student_output             

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

    def __init__(self, img_size=[40, 80, 154, 112], patch_size=1, in_chans=3, num_classes=65536, embed_dim=[768, 192, 384, 96], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, upsample=False, patch_norm=True,
                 use_checkpoint=False, use_deformable_block: bool = True, fused_window_process=False, stage_idx=[0, 1, 2, 3], strides=[1,1, 1, 1], groups=[1, 1, 3, 6], use_pes=[True, True, True, True], 
                 dwc_pes=[True, True, True, True], fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 offset_range_factor=[1, 2, 3, 4], 
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim[0] * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.layers_t = nn.ModuleList()
        ### define a function calling two lists and produce patches and store in a list again used in the basic layer 
        ### in the basic layer, we need a callin function to replace the upsample which calls the next item in a list
        self.patch_embed_4x = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
                norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_2x = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[1],
                norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_1x = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[2],
                norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_11x = PatchEmbed(
                img_size=img_size[3], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[3],
                norm_layer=norm_layer if self.patch_norm else None)
            
        num_patches_4x = self.patch_embed_4x.num_patches
        patches_resolution_4x = self.patch_embed_4x.patches_resolution
        self.patches_resolution_4x = patches_resolution_4x
         
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_4x = nn.Parameter(torch.zeros(1, num_patches_4x, embed_dim))
            trunc_normal_(self.absolute_pos_embed_4x, std=.02)

        num_patches_2x = self.patch_embed_2x.num_patches
        patches_resolution_2x = self.patch_embed_2x.patches_resolution
        self.patches_resolution_2x = patches_resolution_2x
         
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_2x = nn.Parameter(torch.zeros(1, num_patches_2x, embed_dim))
            trunc_normal_(self.absolute_pos_embed_2x, std=.02)

        num_patches_1x = self.patch_embed_1x.num_patches
        patches_resolution_1x = self.patch_embed_1x.patches_resolution
        self.patches_resolution_1x = patches_resolution_1x
         
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_1x = nn.Parameter(torch.zeros(1, num_patches_1x, embed_dim))
            trunc_normal_(self.absolute_pos_embed_1x, std=.02)

        num_patches_11x = self.patch_embed_11x.num_patches
        patches_resolution_11x = self.patch_embed_11x.patches_resolution
        self.patches_resolution_11x = patches_resolution_11x
         
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_11x = nn.Parameter(torch.zeros(1, num_patches_11x, embed_dim))
            trunc_normal_(self.absolute_pos_embed_11x, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # self.patch_dim = [int((x / patch_size)/2**3) for x in img_shape]
        # self.patch_dim = img_size // patch_size

        # build layers
        in_channels = [ 
                384,
                192, 
                384, 
                96
                ]  

            # up_inputs = [
            #     80, 
            #     160
            #     ]  
            # up_dims = [
            #     96, 
            #     48
            #     ]  
            
            
            ### teacher
        # for index in range(self.num_layers):
        # in_channel = in_channels[0]
        # self.layer_4x = BasicLayer(dim=in_channels[0],
        #                                 input_resolution=(patches_resolution_4x[0],
        #                                                 patches_resolution_4x[1]),
        #                                 depth=depths[0],
        #                                 num_heads=num_heads[0],
        #                                 window_size=window_size,
        #                                 stride=strides[0],
        #                                 n_groups=groups[0],
        #                                 offset_range_factor=offset_range_factor[0],
        #                                 use_pe=use_pes[0],
        #                                 stage_idx=stage_idx[0],
        #                                 fixed_pe=fixed_pes[0],
        #                                 no_off=no_offs[0],
        #                                 dwc_pe=dwc_pes[0],
        #                                 mlp_ratio=self.mlp_ratio,
        #                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                 drop=drop_rate, attn_drop=attn_drop_rate,
        #                                 drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
        #                                 norm_layer=norm_layer,
        #                                 # upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
        #                                 use_checkpoint=use_checkpoint,
        #                                 fused_window_process=fused_window_process)

        # self.layer_2x = BasicLayer(dim=in_channels[1],
        #                                 input_resolution=(patches_resolution_2x[0],
        #                                                 patches_resolution_2x[1]),
        #                                 depth=depths[1],
        #                                 num_heads=num_heads[1],
        #                                 window_size=window_size,
        #                                 stride=strides[1],
        #                                 n_groups=groups[1],
        #                                 offset_range_factor=offset_range_factor[1],
        #                                 use_pe=use_pes[1],
        #                                 stage_idx=stage_idx[1],
        #                                 fixed_pe=fixed_pes[1],
        #                                 no_off=no_offs[1],
        #                                 dwc_pe=dwc_pes[1],
        #                                 mlp_ratio=self.mlp_ratio,
        #                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                 drop=drop_rate, attn_drop=attn_drop_rate,
        #                                 drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
        #                                 norm_layer=norm_layer,
        #                                 # upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
        #                                 use_checkpoint=use_checkpoint,
        #                                 fused_window_process=fused_window_process)

        self.layer_1x = BasicLayer(dim=in_channels[2],
                                        input_resolution=(patches_resolution_1x[0],
                                                        patches_resolution_1x[1]),
                                        depth=depths[2],
                                        num_heads=num_heads[2],
                                        window_size=window_size,
                                        stride=strides[2],
                                        n_groups=groups[2],
                                        offset_range_factor=offset_range_factor[2],
                                        use_pe=use_pes[2],
                                        stage_idx=stage_idx[2],
                                        fixed_pe=fixed_pes[2],
                                        no_off=no_offs[2],
                                        dwc_pe=dwc_pes[2],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                        norm_layer=norm_layer,
                                        # upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)

        # self.layer_11x = BasicLayer(dim=in_channels[3],
        #                                 input_resolution=(patches_resolution_11x[0],
        #                                                 patches_resolution_11x[1]),
        #                                 depth=depths[3],
        #                                 num_heads=num_heads[3],
        #                                 window_size=window_size,
        #                                 stride=strides[3],
        #                                 n_groups=groups[3],
        #                                 offset_range_factor=offset_range_factor[3],
        #                                 use_pe=use_pes[3],
        #                                 stage_idx=stage_idx[3],
        #                                 fixed_pe=fixed_pes[3],
        #                                 no_off=no_offs[3],
        #                                 dwc_pe=dwc_pes[3],
        #                                 mlp_ratio=self.mlp_ratio,
        #                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                 drop=drop_rate, attn_drop=attn_drop_rate,
        #                                 drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #                                 norm_layer=norm_layer,
        #                                 # upsample=BilinearUpsample if (i_layer < self.num_layers - 1) else None,
        #                                 use_checkpoint=use_checkpoint,
        #                                 fused_window_process=fused_window_process)

            # self.layers_t.append(layer_t)
            # groups = right_rotate(groups, 3)
            # offset_range_factor = right_rotate(offset_range_factor, 3)
            # num_heads = right_rotate(num_heads, 3)
            # depths = right_rotate(depths, 3)
            # in_channels = right_rotate(in_channels, 3)
        
        self.norm = norm_layer(384)
        # self.norm_last = norm_layer(self.num_features//2**5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(384, num_classes) if num_classes > 0 else nn.Identity()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.out = nn.Conv2d(96, 3, 1, 1, 0)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
  
    def forward_features_teacher(self, s_in, t_in):     
        teacher_output = [] 
        # lr = self.patch_embed_4x(s_in[0])
        # ref = self.patch_embed_4x(t_in[0])
        # if self.ape:
        #     lr = lr + self.absolute_pos_embed_4x
        #     ref = ref + self.absolute_pos_embed_4x
        # lr = self.pos_drop(lr)
        # ref = self.pos_drop(ref)
        # lr, _ = self.layer_4x(lr, ref)
        # teacher_output.append(lr)

        # lr = self.patch_embed_2x(s_in[1])
        # ref = self.patch_embed_2x(t_in[1])
        # if self.ape:
        #     lr = lr + self.absolute_pos_embed_2x
        #     ref = ref + self.absolute_pos_embed_2x
        # lr = self.pos_drop(lr)
        # ref = self.pos_drop(ref)
        # lr, _ = self.layer_2x(lr, ref)
        # teacher_output.append(lr)
        
        lr = self.patch_embed_1x(s_in)
        ref = self.patch_embed_1x(t_in)
        if self.ape:
            lr = lr + self.absolute_pos_embed_1x
            ref = ref + self.absolute_pos_embed_1x
        lr = self.pos_drop(lr)
        ref = self.pos_drop(ref)
        lr, ref = self.layer_1x(lr, ref)
        lr = self.norm(lr)  # B L C
        x = self.avgpool(lr.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        for i in range (4):
            teacher_output.append(x)   


        # lr = self.patch_embed_11x(s_in[3])
        # ref = self.patch_embed_11x(t_in[3])
        # if self.ape:
        #     lr = lr + self.absolute_pos_embed_11x
        #     ref = ref + self.absolute_pos_embed_11x
        # lr = self.pos_drop(lr)
        # ref = self.pos_drop(ref)
        # lr, _ = self.layer_11x(lr, ref)
        # teacher_output.append(lr)

        return teacher_output

    def forward(self, in_t, ref_t):
        teacher_output = self.forward_features_teacher(in_t, ref_t)

        return teacher_output
                    

    def flops(self):
        flops = 0
        flops += self.patch_embed_2x.flops() 
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution_2x[0] * self.patches_resolution_2x[1] // (2 ** self.num_layers)
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