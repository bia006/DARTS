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
from gpytorch.kernels.kernel import Distance
from einops.layers.torch import Rearrange

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
        # self.norm = nn.LayerNorm(in_channel)
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer="gelu", drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = GeLU()
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


def gelu(x):
    """Implementation of the GeLU() activation function.
        For information: OpenAI GPT's GeLU() is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the GeLU() activation function.
        For information: OpenAI GPT's GeLU() is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

ACT2FN = {"gelu": gelu}

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

    def __init__(self, dim, window_size, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, locality_strength,          use_local_init, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.1, fc=None):

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
        self.use_local_init = use_local_init
        self.norm = nn.LayerNorm(dim)
   
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(0)
        self.pos_proj = nn.Linear(3, num_heads)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.conv = nn.Conv2d(num_heads*2, num_heads, 1, 1, 0)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v_add = torch.nn.Parameter(
            torch.FloatTensor(self.dim).uniform_(-0.1, 0.1))
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
        self.linear = nn.Linear(dim*2, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.score_proj = nn.Linear(dim, 1)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1) 

        self.apply(self._init_weights)
        if self.use_local_init:
            self.local_init(locality_strength=self.locality_strength)
        

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  
    
    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength


    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)

        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)

    def get_attention(self, q, k):
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def forward(self, q, k, v, q2, k2, v2, mask=None, mask_heads=None, nb_task_tokens=1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B, N, C = q.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        # q = q[:,:nb_task_tokens].reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        patch_score = (q @ k.transpose(-2, -1)) * self.scale	

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # patch_score = patch_score + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            patch_score = patch_score.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            patch_score = patch_score.view(-1, self.num_heads, N, N)
            patch_score = patch_score.softmax(dim=-1)
        else:
            patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)
        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn1 = self.attn_drop(attn)
        # v = nn.Parameter(torch.ones(B, N, C)).cuda()
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        attn1 = self.norm(self.proj_drop(x))

     #########################   
        # B, N, C = q.shape
        # if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
        #     self.get_rel_indices(N)

        # q = q[:,:nb_task_tokens].reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = k2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        patch_score = (q @ k2.transpose(-2, -1)) * self.scale	

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # patch_score = patch_score + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            patch_score = patch_score.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            patch_score = patch_score.view(-1, self.num_heads, N, N)
        #     # attn = self.softmax(attn)
            patch_score = patch_score.softmax(dim=-1)
        else:
        #     # attn = self.softmax(attn)									
            patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)
        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn2 = self.attn_drop(attn)
        # v = nn.Parameter(torch.ones(B, N, C)).cuda()
        v2 = v2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        attn2 = self.norm(self.proj_drop(x))
        #######
 
        q = self.q(attn1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(attn1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = self.k(attn2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        weighted1 = (q @ k.transpose(-2, -1)) * self.scale
        weighted2 = (q @ k2.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        weighted1 = weighted1 + relative_position_bias.unsqueeze(0)
        # weighted = torch.einsum('b h m c, b h n c -> b h m n', attn1, attn2)
        weighted1 = weighted1.softmax(dim=-1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        weighted2 = weighted2 + relative_position_bias.unsqueeze(0)
        # weighted = torch.einsum('b h m c, b h n c -> b h m n', attn1, attn2)
        weighted2 = weighted2.softmax(dim=-1)

        attn = (1.-torch.sigmoid(gating)) * weighted1 + torch.sigmoid(gating) * weighted2
        # attn = (1.-torch.tanh(gating)) * weighted + torch.tanh(gating) * weighted
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        # v = nn.Parameter(torch.ones(B, N, C)).cuda()
        v = self.v(attn2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        attn = self.proj_drop(x)

        return attn, attn, attn

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


class WindowAttention_SA(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, locality_strength,          use_local_init, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):

        super().__init__()
        ### Standard attn block 
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.n_head_channels = self.head_dim
        self.nc = self.n_head_channels * self.num_heads
        # self.scale = qk_scale or self.head_dim ** -0.5
        self.scale = self.head_dim ** 0.5
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.num_heads // self.n_groups
        self.stride = stride
        self.offset_range_factor = offset_range_factor
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe
        self.use_local_init = use_local_init
   
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads), requires_grad= True)
        self.mu = nn.Parameter((torch.empty( 2, self.num_heads, self.head_dim).normal_(mean = 0.0, std = .5))*torch.tensor([0.,1.])[:, None, None], requires_grad= True)
        self.pi = nn.Parameter(torch.tensor([0.5, 0.5]),requires_grad= True)
        self.dist = Distance()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(0)
        self.pos_proj = nn.Linear(3, num_heads)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.silu = torch.nn.SiLU()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1) 
        # self.register_buffer('pi',0.5*torch.ones(self.num_heads, 2, requires_grad=False))
        self.register_buffer('kk_distance',torch.tensor(0., requires_grad = True))

        self.apply(self._init_weights)       

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  
    
    # def local_init(self, locality_strength=1.):
    #     self.v.weight.data.copy_(torch.eye(self.dim))
    #     locality_distance = 1 #max(1,1/locality_strength**.5)

    #     kernel_size = int(self.num_heads**.5)
    #     center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
    #     for h1 in range(kernel_size):
    #         for h2 in range(kernel_size):
    #             position = h1+kernel_size*h2
    #             self.pos_proj.weight.data[position,2] = -1
    #             self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
    #             self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
    #     self.pos_proj.weight.data *= locality_strength


    # def get_rel_indices(self, num_patches):
    #     img_size = int(num_patches**.5)
    #     rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    #     ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
    #     indx = ind.repeat(img_size,img_size)
    #     indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
    #     indd = indx**2 + indy**2
    #     rel_indices[:,:,:,2] = indd.unsqueeze(0)
    #     rel_indices[:,:,:,1] = indy.unsqueeze(0)
    #     rel_indices[:,:,:,0] = indx.unsqueeze(0)
    #     device = self.qk.weight.device
    #     self.rel_indices = rel_indices.to(device)

    def forward(self, q, k1, k2, v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = q.shape

        # q = q[:,:nb_task_tokens].reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = k1.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = k2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = 0.
        q = q * self.scale
        attn1 = (q @ k1.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn1 = attn1 + relative_position_bias.unsqueeze(0)

        attn2 = (q @ k2.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn2 = attn2 + relative_position_bias.unsqueeze(0)

        # QK1_distance = self.dist._sq_dist(q, k1 - self.mu[0][None, :, None, :], postprocess = False)
        # QK2_distance = self.dist._sq_dist(q, k1 - self.mu[1][None, :, None, :], postprocess = False)
        # dist_min1 = torch.minimum(QK1_distance, QK2_distance)
        # dist_min1 = (-1/(2*self.scale))*dist_min1
        # dist_min1 = dist_min1.softmax(dim=-1)        attn1 = (q @ k1.transpose(-2, -1))
        

        # QK1_distance = self.dist._sq_dist(q, k1 - self.mu[0][None, :, None, :], postprocess        attn1 = (q @ k1.transpose(-2, -1))
        # QK1_distance = self.dist._sq_dist(q, k1 - self.mu[0][None, :, None, :], postprocess

        # QK1_distance = self.dist._sq_dist(q, k2 - self.mu[0][None, :, None, :], postprocess = False)
        # QK2_distance = self.dist._sq_dist(q, k2 - self.mu[1][None, :, None, :], postprocess = False)
        # dist_min2 = torch.minimum(QK1_distance, QK2_distance)
        # dist_min2 = (-1/(2*self.scale))*dist_min2 
        # dist_min2 = dist_min2.softmax(dim=-1)
        
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn1 = attn1.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N)

            attn2 = attn2.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N)
            # attn /= attn.sum(dim=-1).unsqueeze(-1)
            attn1 = self.softmax(attn1)
            attn2 = self.softmax(attn2)
        else:
            attn1 = self.softmax(attn1)
            attn2 = self.softmax(attn2)
            
        gating = (self.gating_param).view(1,-1,1,1) 
        # attn = (1.-torch.sigmoid(gating)) * dist_min1 + torch.sigmoid(gating) * dist_min2
        attn = (1.-torch.sigmoid(gating)) * attn1 + torch.sigmoid(gating) * attn2
                             
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
       
        return x, x, x

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
    

class WindowAttention_MHSA(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, locality_strength,          use_local_init, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):

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
        self.use_local_init = use_local_init
   
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.mu = nn.Parameter((torch.empty( 2, self.num_heads, self.head_dim).normal_(mean = 0.0, std = .5))*torch.tensor([0.,1.])[:, None, None], requires_grad= True)
        self.pi = nn.Parameter(torch.tensor([0.5, 0.5]),requires_grad= True)
        self.dist = Distance()

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
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)
        # if self.use_local_init:
        #     self.local_init(locality_strength=self.locality_strength)
        

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  

    def forward(self, q, k, v, mask=None, mask_heads=None, attn_mask=False, nb_task_tokens=1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        #### MHSA
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k2 = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = 0.

        QK1_distance = self.dist._sq_dist(q, k1 - self.mu[0][None, :, None, :], postprocess = False)
        QK2_distance = self.dist._sq_dist(q, k1 - self.mu[1][None, :, None, :], postprocess = False)
        

        # QK2_distance = (-1/(2*self.scale))*self.dist._sq_dist(q, k2 - self.mu[1][None, :, None, :], postprocess = False) 
        gating = self.gating_param.view(1,-1,1,1)
        attn = torch.clamp((1.-torch.sigmoid(gating)) * QK1_distance, min = 0., max = 1.) + torch.clamp(torch.sigmoid(gating) * QK2_distance, min = 0., max = 1.)
        # attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1] 
        # kk_distance = self.dist._sq_dist(k1, k2, postprocess = False).detach().mean()
        # kk_distance.to(q)*torch.clamp(self.pi, min = 0., max = 1.)[1]
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn/(attn.sum(dim = -1)[:, :, :, None])
             # attn = attn.softmax(dim=-1)
        else:
            # attn = attn.softmax(dim=-1)
            attn = attn/(attn.sum(dim = -1)[:, :, :, None])
         # kk_distance = self.dist._sq_dist(k1, k2, postprocess = False).detach().mean()
        # kk_distance.to(q)
        # self.kk_distance.copy_(kk_distance)
                     
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
       
        return x, x, x

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


class ScaleNorm(nn.Module):
    """See
    https://github.com/lucidrains/reformer-pytorch/blob/a751fe2eb939dcdd81b736b2f67e745dc8472a09/reformer_pytorch/reformer_pytorch.py#L143
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

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
        act_layer (nn.Module, optional): Activation layer. Default: GeLU()
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, locality_strength, use_local_init, no_off, fixed_pe, window_size, shift_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=GeLU(), norm_layer=nn.LayerNorm, fused_window_process=False, fc=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm2 = AdaptiveInstanceNorm(self.dim, self.dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_groups=n_groups, stage_idx=stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe, locality_strength=locality_strength, use_local_init=use_local_init, fc=fc)
        
        self.attn_SA = WindowAttention_SA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_groups=n_groups, stage_idx=stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe, locality_strength=locality_strength, use_local_init=use_local_init, fc=fc)
        
        self.attn_MHSA = WindowAttention_MHSA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_groups=n_groups, stage_idx=stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe, locality_strength=locality_strength, use_local_init=use_local_init, fc=fc)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.linear = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=False)
     
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask1 = None
        attn_mask2 = None
        attn_mask3 = None
        attn_mask4 = None

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
        
            attn_mask4 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask4 = attn_mask4.masked_fill(
                attn_mask4 != 0, float(-100.0)).masked_fill(attn_mask4 == 0, float(0.0))
            
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)
        self.register_buffer("attn_mask3", attn_mask3)
        self.register_buffer("attn_mask4", attn_mask4)
        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process
     
    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ref):
        ### first GPSA
        H, W = self.input_resolution
       
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        shortcut_ref = ref
        x = self.norm1(x)
        ref = self.norm1(ref)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, :].reshape(3, B, H, W, C)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, :], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C)
        else:
            qkv_2 = qkv[:, :, :, :].reshape(3, B, H, W, C )

        qkv_ref = self.qkv(ref).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_ref1 = qkv_ref[:, :, :, : ].reshape(3, B, H, W, C)
        if self.shift_size > 0:
            qkv_ref2 = torch.roll(qkv_ref[:, :, :, :], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C)
        else:
            qkv_ref2 = qkv_ref[:, :, :, :].reshape(3, B, H, W, C)


        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_ref2)

        q3_windows, k3_windows, v3_windows = self.get_window_qkv(qkv_2)
        q4_windows, k4_windows, v4_windows = self.get_window_qkv(qkv_ref1)

        x1, ref1, _ = self.attn_SA(q1_windows, k1_windows, k2_windows, v1_windows, self.attn_mask)
        x2, ref2, _ = self.attn_SA(q1_windows, k1_windows, k2_windows, v1_windows, self.attn_mask)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C), self.window_size, H, W) 
        ref1 = window_reverse(ref1.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        ref2 = window_reverse(ref2.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)      

        x3, ref3, _ = self.attn_SA(q3_windows, k3_windows, k4_windows, v3_windows, self.attn_mask)
        x4, ref4, _ = self.attn_SA(q3_windows, k3_windows, k4_windows, v3_windows, self.attn_mask)
        
        x3 = window_reverse(x3.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        x4 = window_reverse(x4.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        ref3 = window_reverse(ref3.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        ref4 = window_reverse(ref4.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)

        # cyclic shift
        if self.shift_size > 0:
            x3 = torch.roll(x3, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            ref3 = torch.roll(ref3, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            ref2 = torch.roll(ref2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x3 = x3
            ref3 = ref3
            x2 = x2  
            ref2 = ref2         

        x = torch.cat([x1.reshape(B, H * W, C), x2.reshape(B, H * W, C)], dim=2)
        x = self.linear(x)

        x_ = torch.cat([x3.reshape(B, H * W, C), x4.reshape(B, H * W, C)], dim=2)      
        x_ = self.linear(x_)
        x = self.linear(torch.cat((x, x_), 2))

        x = shortcut + x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        # x_GPSA = x

        # x = torch.tanh(self.linear(torch.cat((x, shortcut), 2)))

        ######
        ## 1st MHSA
        # H, W = self.input_resolution
       
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # shortcut = x
        # x = self.norm1(x)
        # # ref = self.norm1(ref)

        # qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        # qkv_1 = qkv[:, :, :, :].reshape(3, B, H, W, C)
        # if self.shift_size > 0:
        #     qkv_2 = torch.roll(qkv[:, :, :, :], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C)
        # else:
        #     qkv_2 = qkv[:, :, :, :].reshape(3, B, H, W, C )

        # q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        # q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        # x1, _, _ = self.attn_MHSA(q1_windows, k1_windows, v1_windows, self.attn_mask1)
        # x2, _, _ = self.attn_MHSA(q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        # x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)
        # x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C), self.window_size, H, W)      

        # # cyclic shift
        # if self.shift_size > 0:
        #     #x3 = torch.roll(x3, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        #     x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     #x3 = x3
        #     x2 = x2           

        # x = torch.cat([x1.reshape(B, H * W, C), x2.reshape(B, H * W, C)], dim=2)
        # x = self.linear(x)     
        # # x = self.linear(torch.cat((x, x_GPSA), 2))  

        # x = shortcut + x + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm1(x)))

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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, n_groups, stage_idx, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, locality_strength, use_local_init,               mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False,
                 fused_window_process=False, fc=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RefSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 n_groups=n_groups, stage_idx= stage_idx, stride=stride, offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe, locality_strength=locality_strength,
                                 use_local_init=use_local_init,
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsample = nn.PixelShuffle(2)
        # self.conv = nn.Conv2d(dim, dim*4, 3, 1, 1)
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

        # # Add SPE    
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
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GeLU() -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GeLU() -> Linear; Permute back
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
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = GeLU()
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


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResidualBlock_(nn.Module):
    """Residual block with BN.
    It has a style of:
        ---Conv-BN-ReLU-Conv-BN-+-
         |______________________|
    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        bn_affine (bool): Whether to use affine in BN layers. Default: True.
    """

    def __init__(self, nf=192, bn_affine=True):
        super(ResidualBlock_, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(nf, affine=True)
        self.conv2 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.bn2 = nn.BatchNorm2d(nf, affine=True)
        self.relu = nn.ReLU(inplace=True)

        default_init_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return identity + out

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
    
class ResidualBlock(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ResidualBlock, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockwithBN, n_blocks, nf=nf)

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

    def __init__(self, img_size=40, patch_size=4, in_chans=6, in_channel=3, num_classes=65536, embed_dim=48, depths=[2, 2, 2, 2], num_heads=[12, 24, 24, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, upsample=False, patch_norm=True,
                 use_checkpoint=False, use_deformable_block: bool = True, fused_window_process=False, stage_idx=[0, 1, 2, 3, 4], strides=[1, 1, 1, 1, 1], groups=[6, 6, 6, 6, 6], use_pes=[False, False, False, False, False], 
                 dwc_pes=[False, False, False, False, False], fixed_pes=[False, False, False, False, False],
                 no_offs=[False, False, False, False, False],
                 offset_range_factor=[1, 2, 3, 4, 5], locality_strength=[1., 1., 1., 1.], use_local_init=[True, True, True, True], fc=None,
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
            96, #160
            96, 
            96, 
            96
            ] 
        
        self.layers = nn.ModuleList()
        self.layers_ref = nn.ModuleList()
        self.layers_ref_g = nn.ModuleList()
                
        ### student
        for i_layer in range(self.num_layers - 3):
        # for i_layer in range(self.num_layers - 2):
            in_channel = in_channels[1]
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
                                locality_strength=locality_strength[i_layer],
                                use_local_init=use_local_init[i_layer],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                norm_layer=norm_layer,
                                upsample=BilinearUpsample if (i_layer < self.num_layers - 3) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process)
            self.layers.append(layer)

        for i_layer_ref in range(self.num_layers - 3):
            in_channel = in_channels[0]
            layer_ref = BasicLayer(dim=in_channel,
                                        # input_resolution=(patches_resolution[0] * (2 ** i_layer_ref),
                                        #                 patches_resolution[1] * (2 ** i_layer_ref)),
                                        input_resolution=(img_size* (2 ** i_layer_ref), img_size* (2 ** i_layer_ref)),
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
                                        locality_strength=locality_strength[i_layer_ref],
                                        use_local_init=use_local_init[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        upsample=BilinearUpsample if (i_layer < self.num_layers - 3) else None,
                                        use_checkpoint=use_checkpoint,
                                        fused_window_process=fused_window_process)
            self.layers_ref.append(layer_ref)

        for i_layer_ref in range(self.num_layers - 3):
            in_channel = in_channels[2]
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
                                        locality_strength=locality_strength[i_layer_ref],
                                        use_local_init=use_local_init[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
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
        patch_dim = 3 * 8 * 8
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 8, p2 = 8),
            nn.LayerNorm(in_channels[-1]),
            nn.Linear(in_channels[-1], in_channels[-1]),
            nn.LayerNorm(in_channels[-1]),
        )

        self.res = ResidualBlock(
            in_nc=3, out_nc=3, nf=in_channels[-1], n_blocks=16)
        self.norm = nn.LayerNorm(in_channels[-1]) # final norm layer
        self.dwconv = nn.Conv2d(in_channels[-1]*2, in_channels[-1]*2, kernel_size=7, stride=1, padding=3) # depthwise conv
        self.act = GeLU()
        self.proj = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=4, stride=4)
        self.proj_1 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1, stride=1)
        self.proj_2 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels[-1], 4*in_channels[-1], 1, 1, 0)
        self.upsample = nn.PixelShuffle(2)

        self.patch_embed = nn.Conv2d(3, in_channels[-1], kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(in_channels[-1]//2, in_channels[-1], 3, 1, 1)
        self.conv_down = nn.Conv2d(in_channels[-1], in_channels[-1], 3, 2, 1)
        self.conv_ref = nn.Conv2d(in_channels[-1]*2, in_channels[-1], 3, 1, 1)
        self.conv_ref1 = nn.Conv2d(in_channels[-1]*2, in_channels[-1], 3, 1, 1)
        self.conv_ref2 = nn.Conv2d(in_channels[-1]*2, in_channels[-1], 3, 1, 1)
        self.conv_up = nn.Conv2d(in_channels[-1], in_channels[-1]*4, 3, 1, 1)
        self.conv_31 = nn.Conv2d(256, in_channels[-1], 3, 1, 1)
        self.conv_21 = nn.Conv2d(128, in_channels[-1], 3, 1, 1)
        self.conv_11 = nn.Conv2d(64, in_channels[-1], 3, 1, 1)
        # self.norm_last = norm_layer(self.num_features//2**5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels[-1], num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Linear(in_channels[-1], num_classes) 
        self.head_in = nn.Conv2d(3, in_channels[-1], 3, 1, 1) 
        self.head_1 = nn.Conv2d(352, in_channels[-1], 1, 1, 0)
        self.head_2 = nn.Conv2d(224, in_channels[-1], 1, 1, 0)
        self.head_3 = nn.Conv2d(131, in_channels[-1], 1, 1, 0)
        self.head_4 = nn.Conv2d(160, in_channels[-1], 1, 1, 0)
        self.out = nn.Conv2d(in_channels[-1], 3, 3, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=in_channels[-1], n_blocks=16)
        
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
       
        ref1 = self.patch_embed(ref1)
        ref = self.patch_embed(ref)
        ref_g = self.patch_embed(ref_g)
                     
        base = F.interpolate(shortcut, None, 4, 'bilinear', False)

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref1 = ref1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        lr = self.to_patch_embedding(lr)
        ref1 = self.to_patch_embedding(ref1)
                
        for layer in self.layers_ref:
            lr, ref1 = layer(lr, ref1)  
     
        lr_1 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*2, shortcut.shape[3]*2).contiguous() #240
        # lr_1 = self.conv_ref2(torch.cat((lr_1, ref), 1))

        B, C, H, W = lr_1.shape
        lr = lr_1.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref = ref.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        # lr = self.to_patch_embedding(lr)
        ref = self.to_patch_embedding(ref)
        for layer in self.layers:
            lr, ref = layer(lr, ref)   

        lr_2 = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous() #192

        # lr_2 = self.conv_ref(torch.cat((lr_2, ref_g), 1))

        B, C, H, W = lr_2.shape
        lr = lr_2.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_g = ref_g.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  

        # lr = self.to_patch_embedding(lr)
        ref_g = self.to_patch_embedding(ref_g)   
       
        for layer in self.layers_ref_g:
            lr, ref_g = layer(lr, ref_g) 
   
        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous() 
        
        out = self.out(lr ) + self.bias + base
       
        return out, out

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

    def __init__(self, img_size=40, patch_size=4, in_chans=6, in_channel=3, num_classes=65536, embed_dim=48, depths=[2, 2, 2, 2], num_heads=[24, 16, 12, 6],
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
                                # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                drop_path=dpr[i_layer],
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
                                        drop_path=dpr[i_layer_ref],
                                        # drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
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
                                        drop_path=dpr[i_layer_ref],
                                        # drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
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
        self.mlp = Mlp(in_channels[-1], in_channels[-1], act_layer=GeLU(), drop=0.)
        self.act = GeLU()
        self.proj = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=4, stride=4)
        self.proj_1 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1, stride=1)
        self.proj_2 = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels[-1], 4 * in_channels[-1], 1, 1, 0)
        self.upsample = nn.PixelShuffle(2)
        
        self.conv1 = nn.Conv2d(in_channels[-1]*2, in_channels[-1]*4, 1, 1, 0)
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
        self.out = nn.Conv2d(in_channels[0], 3, 3, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=384, n_blocks=8)
        
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
        lr = self.conv1(torch.cat((lr, ref_g), 1))
        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_g = ref_g.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)     
       
        for layer in self.layers_ref_g:
            lr, ref_g = layer(lr, ref_g) 
            # lr = self.mlp(lr) + lr
    
        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*4, shortcut.shape[3]*4).contiguous() 
       
        B, C, H, W = lr.shape
        x_ref = lr.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        x = self.norm(x_ref)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        # x.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False
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
