# from collections import OrderedDict
# import torch
# import math
# import numpy as np
# from torch import nn
# from torch.nn import functional as F
# from torch.nn.utils import spectral_norm
# from torch.nn.modules.sparse import Embedding
# from .arch_util import srntt_init_weights
# import functools


# from .CustomLayers import (EqualizedConv2d, EqualizedLinear,
#                                  PixelNormLayer, Truncation, BlurLayer, View, StddevLayer)

# class DiscriminatorTop(nn.Sequential):
#     def __init__(self,
#                  mbstd_group_size,
#                  mbstd_num_features,
#                  in_channels,
#                  intermediate_channels,
#                  gain, use_wscale,
#                  activation_layer,
#                  resolution=4,
#                  in_channels2=None,
#                  output_features=1,
#                  last_gain=1):
#         """
#         :param mbstd_group_size:
#         :param mbstd_num_features:
#         :param in_channels:
#         :param intermediate_channels:
#         :param gain:
#         :param use_wscale:
#         :param activation_layer:
#         :param resolution:
#         :param in_channels2:
#         :param output_features:
#         :param last_gain:
#         """

#         layers = []
#         if mbstd_group_size > 1:
#             layers.append(('stddev_layer', StddevLayer(mbstd_group_size, mbstd_num_features)))

#         if in_channels2 is None:
#             in_channels2 = in_channels

#         layers.append(('conv', EqualizedConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
#                                                gain=gain, use_wscale=use_wscale)))
#         layers.append(('act0', activation_layer))
#         layers.append(('view', View(-1)))
#         layers.append(('dense0', EqualizedLinear(in_channels2 * resolution * resolution, intermediate_channels,
#                                                  gain=gain, use_wscale=use_wscale)))
#         layers.append(('act1', activation_layer))
#         layers.append(('dense1', EqualizedLinear(intermediate_channels, output_features,
#                                                  gain=last_gain, use_wscale=use_wscale)))

#         super().__init__(OrderedDict(layers))


# class DiscriminatorBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
#         super().__init__(OrderedDict([
#             ('conv0', EqualizedConv2d(in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)),
#             # out channels nf(res-1)
#             ('act0', activation_layer),
#             ('blur', BlurLayer(kernel=blur_kernel)),
#             ('conv1_down', EqualizedConv2d(in_channels, out_channels, kernel_size=3,
#                                            gain=gain, use_wscale=use_wscale, downscale=True)),
#             ('act1', activation_layer)]))


# class ImageDiscriminator(nn.Module):

#     def __init__(self, resolution=1024, num_channels=3, conditional=False,
#                  n_classes=0, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
#                  nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4,
#                  mbstd_num_features=1, blur_filter=None, structure='linear',
#                  **kwargs):
#         """
#         Discriminator used in the StyleGAN paper.
#         :param num_channels: Number of input color channels. Overridden based on dataset.
#         :param resolution: Input resolution. Overridden based on dataset.
#         # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
#         :param fmap_base: Overall multiplier for the number of feature maps.
#         :param fmap_decay: log2 feature map reduction when doubling the resolution.
#         :param fmap_max: Maximum number of feature maps in any layer.
#         :param nonlinearity: Activation function: 'relu', 'lrelu'
#         :param use_wscale: Enable equalized learning rate?
#         :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
#         :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
#         :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
#         :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
#         :param kwargs: Ignore unrecognized keyword args.
#         """
#         super(ImageDiscriminator, self).__init__()

#         if conditional:
#             assert n_classes > 0, "Conditional Discriminator requires n_class > 0"
#             # self.embedding = nn.Embedding(n_classes, num_channels * resolution ** 2)
#             num_channels *= 2
#             self.embeddings = []

#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

#         self.conditional = conditional
#         self.mbstd_num_features = mbstd_num_features
#         self.mbstd_group_size = mbstd_group_size
#         self.structure = structure
#         # if blur_filter is None:
#         #     blur_filter = [1, 2, 1]

#         resolution_log2 = int(np.log2(resolution))
#         assert resolution == 2 ** resolution_log2 and resolution >= 4
#         self.depth = resolution_log2 - 1

#         act, gain = {'relu': (torch.relu, np.sqrt(2)),
#                      'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

#         # create the remaining layers
#         blocks = []
#         from_rgb = []
#         for res in range(resolution_log2, 2, -1):
#             # name = '{s}x{s}'.format(s=2 ** res)
#             blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
#                                              gain=gain, use_wscale=use_wscale, activation_layer=act,
#                                              blur_kernel=blur_filter))
#             # create the fromRGB layers for various inputs:
#             from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
#                                             gain=gain, use_wscale=use_wscale))
#             # Create embeddings for various inputs:
#             if conditional:
#                 r = 2 ** (res)
#                 self.embeddings.append(
#                     Embedding(n_classes, (num_channels // 2) * r * r))

#         if self.conditional:
#             self.embeddings.append(nn.Embedding(
#                 n_classes, (num_channels // 2) * 4 * 4))
#             self.embeddings = nn.ModuleList(self.embeddings)

#         self.blocks = nn.ModuleList(blocks)

#         # Building the final block.
#         self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
#                                             in_channels=nf(2), intermediate_channels=nf(2),
#                                             gain=gain, use_wscale=use_wscale, activation_layer=act)
#         from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
#                                         gain=gain, use_wscale=use_wscale))
#         self.from_rgb = nn.ModuleList(from_rgb)

#         # register the temporary downSampler
#         self.temporaryDownsampler = nn.AvgPool2d(2)

#     def forward(self, images_in, depth=4, alpha=1., labels_in=None):
#         """
#         :param images_in: First input: Images [mini_batch, channel, height, width].
#         :param labels_in: Second input: Labels [mini_batch, label_size].
#         :param depth: current height of operation (Progressive GAN)
#         :param alpha: current value of alpha for fade-in
#         :return:
#         """
        
#         assert depth < self.depth, "Requested output depth cannot be produced"

#         if self.conditional:
#             assert labels_in is not None, "Conditional Discriminator requires labels"
#         # print(embedding_in.shape, images_in.shape)
#         # exit(0)
#         # print(self.embeddings)
#         # exit(0)
#         if self.structure == 'fixed':
#             if self.conditional:
#                 embedding_in = self.embeddings[0](labels_in)
#                 embedding_in = embedding_in.view(images_in.shape[0], -1,
#                                                  images_in.shape[2],
#                                                  images_in.shape[3])
#                 images_in = torch.cat([images_in, embedding_in], dim=1)
#             x = self.from_rgb[0](images_in)
#             for i, block in enumerate(self.blocks):
#                 x = block(x)
#             scores_out = self.final_block(x)
            
#         elif self.structure == 'linear':
#             if depth > 0:
#                 if self.conditional:
#                     embedding_in = self.embeddings[self.depth -
#                                                    depth - 1](labels_in)
#                     embedding_in = embedding_in.view(images_in.shape[0], -1,
#                                                      images_in.shape[2],
#                                                      images_in.shape[3])
#                     images_in = torch.cat([images_in, embedding_in], dim=1)
                    
#                 residual = self.from_rgb[self.depth -
#                                          depth](self.temporaryDownsampler(images_in))
#                 straight = self.blocks[self.depth - depth -
#                                        1](self.from_rgb[self.depth - depth - 1](images_in))
#                 x = (alpha * straight) + ((1 - alpha) * residual)

#                 for block in self.blocks[(self.depth - depth):]:
#                     x = block(x)
#             else:
#                 if self.conditional:
#                     embedding_in = self.embeddings[-1](labels_in)
#                     embedding_in = embedding_in.view(images_in.shape[0], -1,
#                                                      images_in.shape[2],
#                                                      images_in.shape[3])
#                     images_in = torch.cat([images_in, embedding_in], dim=1)
#                 x = self.from_rgb[-1](images_in)
                    
#             # scores_out = self.final_block(x)
#         else:
#             raise KeyError("Unknown structure: ", self.structure)

#         return x

# class ImageDiscriminator(nn.Module):

#     def __init__(self, in_nc=3, ndf=32):
#         super(ImageDiscriminator, self).__init__()

#         def conv_block(in_channels, out_channels):
#             block = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#                 nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(out_channels, out_channels, 3, 2, 1),
#                 nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True))
#             return block

#         self.conv_block1 = conv_block(in_nc, ndf)
#         self.conv_block2 = conv_block(ndf, ndf * 2)
#         self.conv_block3 = conv_block(ndf * 2, ndf * 4)
#         self.conv_block4 = conv_block(ndf * 4, ndf * 8)
#         self.conv_block5 = conv_block(ndf * 8, ndf * 16)

#         self.out_block = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Conv2d(ndf * 16, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1), nn.Sigmoid())

#         srntt_init_weights(self, init_type='normal', init_gain=0.02)

#     def forward(self, x):
#         fea = self.conv_block1(x)
#         fea = self.conv_block2(fea)
#         fea = self.conv_block3(fea)
#         fea = self.conv_block4(fea)
#         fea = self.conv_block5(fea)

#         out = self.out_block(fea)

#         return out

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

import math

import torch
from .op import FusedLeakyReLU, upfirdn2d
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .CustomLayers import (Blur, Downsample, EqualConv2d, EqualLinear,
                                 ScaledLeakyReLU)


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        sn=False
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        if sn:
            # Not use equal conv2d when apply SN
            layers.append(
                spectral_norm(nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                ))
            )
        else:
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, sn=sn)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, sn=sn)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
    

class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ConvLayer(3 * 4, out_channel, 1, sn=sn)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self, size=256, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], sn=sn))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        if sn:
            self.final_linear = nn.Sequential(
                spectral_norm(nn.Linear(2048, 512)),
                FusedLeakyReLU(512),
                spectral_norm(nn.Linear(512, 1)),
        )
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(2048, 512, activation='fused_lrelu'),
                EqualLinear(512, 1),
            )

    def forward(self, input):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out