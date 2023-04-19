import logging
from einops import rearrange
import numpy as np


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.distributed as dist

from .archs.vgg_arch import VGGFeatureExtractor
from .loss_utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']

logger = logging.getLogger('base')


@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt((pred - target)**2 + eps)
    

def extract_feature_map(io_dict, feature_map_config):
    io_type = feature_map_config['io']
    module_path = feature_map_config['path']
    if isinstance(module_path,(tuple,list)):
        feat=[]
        for path in module_path:
            feat.append(io_dict[path][io_type])
        return feat
    else:
        return io_dict[module_path][io_type]

def mf_loss(F_s, F_t, K, w_sample=1.0, w_patch=1.0, w_rand=1.0, max_patch_num=0):
        losses = [[], [], []]  # loss_mf_sample, loss_mf_patch, loss_mf_rand
        loss_mf_patch, loss_mf_sample, loss_mf_rand = layer_mf_loss(
                F_s, F_t)
        losses[0].append(w_sample * loss_mf_sample)
        losses[1].append(w_patch * loss_mf_patch)
        losses[2].append(w_rand * loss_mf_rand)

        loss_mf_sample = sum(losses[0]) / len(losses[0])
        loss_mf_patch = sum(losses[1]) / len(losses[1])
        loss_mf_rand = sum(losses[2]) / len(losses[2])

        return loss_mf_sample, loss_mf_patch, loss_mf_rand


def layer_mf_loss(F_t, F_s, K=1):
    # normalize at feature dim
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)

    # manifold loss among different patches (intra-sample)
    M_s = F_s.bmm(F_s.transpose(-1, -2))
    M_t = F_t.bmm(F_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_intra = (M_diff * M_diff).mean()

    # manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))
    M_diff = M_t - M_s
    loss_inter = (M_diff * M_diff).mean()

    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    loss_mf_rand = (M_diff * M_diff).mean()
    return loss_intra, loss_inter, loss_mf_rand


def merge(x, max_patch_num=196):
    B, P, C = x.shape
    if P <= max_patch_num:
        return x
    n = int(P ** (1/2))  # original patch num at each dim
    m = int(max_patch_num ** (1/2))  # target patch num at each dim
    merge_num = n // m  # merge every (merge_num x merge_num) adjacent patches
    x = x.view(B, m, merge_num, m, merge_num, C)
    merged = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, m * m, -1)
    return merged


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)

def inter_class_realtion(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()

def intra_class_realtion(y_s, y_t):
    return inter_class_realtion(y_s.transpose(0,1), y_t.transpose(0,1))

class DISTLoss(nn.Module):
    def __init__(self, beta, gamma):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        totalLoss = 0
        for s, t in zip(z_s, z_t):
            y_s = s.softmax(dim=1)
            y_t = t.softmax(dim=1)
            inter_loss = inter_class_realtion(y_s, y_t)
            intra_loss = intra_class_realtion(y_s, y_t)
            kd_loss = self.beta * inter_loss + self.gamma * intra_loss
            totalLoss += kd_loss
        return totalLoss

class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, alpha, beta, gamma, w_sample, w_patch, w_rand, K, nepochs, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, student_temp, center_momentum, base_criterion = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.base_criterion = base_criterion
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.w_sample = w_sample
        self.w_patch = w_patch
        self.w_rand = w_rand
        self.K = K
        self.nepochs = nepochs
        self.out_dim = out_dim
        self.ncrops = ncrops
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_outs, student_outs, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        for i, (student_output, teacher_output) in enumerate(zip(student_outs, teacher_outs)):       
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.ncrops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(2)

            total_loss = 0
            n_loss_terms = 0
            kd_total = 0
            totalLoss = 0
            for iq, (q, v) in enumerate(zip(teacher_out, student_out)):
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

            # y_s = v.softmax(dim=1)
            # y_t = q.softmax(dim=1)
            # inter_loss = inter_class_realtion(y_s, y_t)
            # intra_loss = intra_class_realtion(y_s, y_t)
            # kd_loss = self.beta * inter_loss + self.gamma * intra_loss  
            # kd_total += kd_loss            

            total_loss /= n_loss_terms
            # totalLoss += total_loss
            # total_loss += kd_total
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
            """
            Update center used for teacher output.
            """
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            # dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * 4)
            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOLoss(nn.Module):
    def __init__(self, nepochs, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, student_temp, center_momentum):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction)


# class MaskedTVLoss(L1Loss):

#     def __init__(self, loss_weight=1.0):
#         super(MaskedTVLoss, self).__init__(loss_weight=loss_weight)

#     def forward(self, pred, mask=None):
#         y_diff = super(MaskedTVLoss, self).forward(
#             pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
#         x_diff = super(MaskedTVLoss, self).forward(
#             pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

#         loss = x_diff + y_diff

#         return loss

class MaskedTVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(MaskedTVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4 feature
            layer (before relu5_4) will be extracted with weight 1.0 in
            calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0,
                 norm_img=True,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class PerceptualLossMultiInputs(PerceptualLoss):
    """Perceptual loss with multiple inputs images.

    Args:
        x (Tensor): Input tensor with shape (B, N, C, H, W), where N indicates
            number of images.
        gt (Tensor): GT tensor with shape (B, N, C, H, W).

    Returns:
        list[Tensor]: total perceptual loss and total style loss.
    """

    def forward(self, x, gt):
        assert x.size() == gt.size(
        ), 'The sizes of input and GT should be the same.'

        total_percep_loss, total_style_loss = 0, 0
        for i in range(x.size(1)):
            percep_loss, style_loss = super(PerceptualLossMultiInputs,
                                            self).forward(
                                                x[:, i, :, :, :],
                                                gt[:, i, :, :, :])
            if percep_loss is None:
                total_percep_loss = None
            else:
                total_percep_loss += percep_loss
            if style_loss is None:
                total_style_loss = None
            else:
                total_style_loss += style_loss

        return total_percep_loss, total_style_loss


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the targe is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


def gradient_penalty_loss(discriminator, real_data, fake_data, mask=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpaitting. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1)).cuda()

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gradients_penalty


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for wgan-gp.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.):
        super(GradientPenaltyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, discriminator, real_data, fake_data, mask=None):
        """
        Args:
            discriminator (nn.Module): Network for the discriminator.
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            mask (Tensor): Masks for inpaitting. Default: None.

        Returns:
            Tensor: Loss.
        """
        loss = gradient_penalty_loss(
            discriminator, real_data, fake_data, mask=mask)

        return loss * self.loss_weight


class Textureloss(nn.Module):
    """ Define Texture Loss.

    Args:
        use_weights (bool): If True, the weights computed in swapping will be
            used to scale the features.
            Default: False
        loss_weight (float): Loss weight. Default: 1.0.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        layer_weights (dict): The weight for each layer of vgg feature.
            Defalut: {'relu1_1': 1.0, 'relu2_1': 1.0, 'relu3_1': 1.0}
        use_input_norm (bool): If True, normalize the input image.
            Default: True.
    """

    def __init__(self,
                 use_weights=True,
                 loss_weight=1.0,
                 vgg_type='vgg19',
                 layer_weights={
                     'relu1_1': 1.0,
                     'relu2_1': 1.0,
                     'relu3_1': 1.0
                 },
                 use_input_norm=True):
        super(Textureloss, self).__init__()
        self.use_weights = use_weights
        self.loss_weight = loss_weight

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

    def gram_matrix(self, features):
        n, c, h, w = features.size()
        feat_reshaped = features.view(n, c, -1)

        # Use torch.bmm for batch multiplication of matrices
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

        return gram

    def forward(self, x, maps, weights=0):
        """
        Args:
            x (Tensor): The input for the loss module.
            maps (Tensor): The maps generated by swap module.
            weights (bool): The weights generated by swap module. The weights
                are used for scale the maps.

        Returns:
            Tensor: Texture Loss value.
        """
        input_size = x.shape[-1]
        x_features = self.vgg(x)

        losses = 0.0
        if self.use_weights:
            if not isinstance(weights, dict):
                weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
        for k in x_features.keys():
            if self.use_weights:
                # adjust the scale according to the name of layer
                if k == 'relu3_1':
                    idx = 0
                    div_num = 256
                elif k == 'relu2_1':
                    idx = 1
                    div_num = 512
                elif k == 'relu1_1':
                    idx = 2
                    div_num = 1024
                else:
                    raise NotImplementedError

                if isinstance(weights, dict):
                    weights_scaled = F.pad(
                        weights[k], (1, 1, 1, 1), mode='replicate')
                else:
                    weights_scaled = F.interpolate(weights, None, 2**idx,
                                                   'bicubic', True)

                # compute coefficients
                # TODO: the input range of tensorflow and pytorch are different,
                # check the values of a and b
                coeff = weights_scaled * (-20.) + .65
                coeff = torch.sigmoid(coeff)

                # weighting features and swapped maps
                maps[k] = maps[k] * coeff
                x_features[k] = x_features[k] * coeff

            # TODO: think about why 4 and **2
            losses += torch.norm(
                self.gram_matrix(x_features[k]) -
                self.gram_matrix(maps[k])) / 4. / (
                    (input_size * input_size * div_num)**2)

        losses = losses / 3.

        return losses * self.loss_weight


class TextureLoss(torch.nn.Module):
    def __init__(self,
                 use_weights=False,
                 loss_weight=1.0,
                 resize=True,
                 vgg_type='vgg19',
                 layer_weights={
                     'relu1_1': 1.0,
                     'relu2_1': 1.0,
                     'relu3_1': 1.0
                 },
                 use_input_norm=True):
        self.use_weights = use_weights
        self.texture_weight = loss_weight
        super(TextureLoss, self).__init__()
        
        # blocks = []
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # for bl in blocks:
        #     for p in bl.parameters():
        #         p.requires_grad = False
        # self.blocks = torch.nn.ModuleList(blocks)
        self.criterion = nn.MSELoss()
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        if self.texture_weight > 0:
            for x_feat, y_feat in zip (input, target):
                loss += self.criterion(x_feat, y_feat.detach()) * self.texture_weight

        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     y = block(y)
        #     if i in feature_layers:
        #         loss += torch.nn.functional.l1_loss(x, y)
        #     if i in style_layers:
        #         act_x = x.reshape(x.shape[0], x.shape[1], -1)
        #         act_y = y.reshape(y.shape[0], y.shape[1], -1)
        #         gram_x = act_x @ act_x.permute(0, 2, 1)
        #         gram_y = act_y @ act_y.permute(0, 2, 1)
        #         loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class MapLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4 feature
            layer (before relu5_4) will be extracted with weight 1.0 in
            calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self, vgg_type='vgg19', map_weight=1.0, criterion='l1'):
        super(MapLoss, self).__init__()
        self.map_weight = map_weight
        self.vgg = VGGFeatureExtractor(
            layer_name_list=['relu3_1', 'relu2_1', 'relu1_1'],
            vgg_type=vgg_type)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, swapped_features, gt):
        # extract vgg features
        gt_features = self.vgg(gt.detach())

        # calculate loss loss
        map_loss = 0
        for k in gt_features.keys():
            if self.criterion_type == 'fro':
                map_loss += torch.norm(
                    swapped_features[k] - gt_features[k], p='fro')
            else:
                map_loss += self.criterion(swapped_features[k], gt_features[k])
        map_loss *= self.map_weight

        return map_loss
