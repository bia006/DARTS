import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn import functional as F
from torch import autograd, nn, optim
from torchvision.models import vgg19
from torch.optim import lr_scheduler

import mmsr.models.networks as networks
import mmsr.utils.metrics as metrics
from mmsr.utils import ProgressBar, tensor2img

from .sr_model import SRModel
from .CR_DiffAug_file import CR_DiffAug

loss_module = importlib.import_module('mmsr.models.losses')
logger = logging.getLogger('base')

def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class SwinTransformerModel(SRModel):

    def __init__(self, opt):
        super(SwinTransformerModel, self).__init__(opt)

        # net_map does not have any trainable parameters.
        # self.net_map = networks.define_net_map(opt)
        # self.net_map = self.model_to_device(self.net_map)
        # define network for feature extraction
        # self.net_extractor = networks.define_net_extractor(opt)
        # self.net_extractor = self.model_to_device(self.net_extractor)
        # self.print_network(self.net_extractor)
        # # load pretrained feature extractor
        # load_path = self.opt['path'].get('pretrain_model_feature_extractor',
        #                                  None)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
            
        if self.is_train:
            self.net_g.train()

            # optimizers
            train_opt = self.opt['train']
            
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
           
            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g
                }], 
                lr=train_opt['lr_g'],
                weight_decay=weight_decay_g,
                betas=train_opt['beta_g'])

            self.optimizers.append(self.optimizer_g)

            self.scheduler_g = lr_scheduler.OneCycleLR(
                        self.optimizer_g,
                        max_lr=train_opt['lr_g'],
                        pct_start=train_opt['warmup'] / train_opt['num_train_steps'],
                        anneal_strategy=train_opt['lr_decay'],
                        total_steps=train_opt['num_train_steps'])
            # self.scheduler_d = lr_scheduler.OneCycleLR(
            #             self.optimizer_d,
            #             max_lr=train_opt['lr_d'],
            #             pct_start=train_opt['warmup'] / train_opt['num_train_steps'],
            #             anneal_strategy=train_opt['lr_decay'],
            #             total_steps=train_opt['num_train_steps'])

    def init_training_settings(self):
        train_opt = self.opt['train']

        if self.opt.get('network_d', None):
            # define network net_d
            self.net_d = networks.define_net_d(self.opt)
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_model_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path']['strict_load'])
        else:
            logger.info('No discriminator.')
            self.net_d = None

        if self.net_d:
            self.net_d.train()

        # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            cri_perceptual_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None

        if train_opt.get('style_opt', None):
            cri_style_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_style = cri_style_cls(**train_opt['style_opt']).to(
                self.device)
        else:
            logger.info('Remove style loss.')
            self.cri_style = None

        if train_opt.get('texture_opt', None):
            cri_texture_cls = getattr(loss_module, 'Textureloss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('gan_type', None):
            cri_gan_cls = getattr(loss_module, 'GANLoss')
            self.cri_gan = cri_gan_cls(
                train_opt['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_weight']).to(self.device)

            if train_opt['grad_penalty_weight'] > 0:
                cri_grad_penalty_cls = getattr(loss_module,
                                               'GradientPenaltyLoss')
                self.cri_grad_penalty = cri_grad_penalty_cls(
                    loss_weight=train_opt['grad_penalty_weight']).to(
                        self.device)
            else:
                logger.info('Remove gradient penalty.')
                self.cri_grad_penalty = None
        else:
            logger.info('Remove GAN loss.')
            self.cri_gan = None

        # we need to train the net_g with only pixel loss for several steps
        self.net_g_pretrain_steps = train_opt['net_g_pretrain_steps']
        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device) # LR/ gt_down
        self.img_in_up = data['img_in_up'].to(self.device) # LR_bicubic_up
        self.img_ref = data['img_ref'].to(self.device) # ref HR
        self.img_ref_lq = data['img_ref_lq'].to(self.device) # ref HR downsampled
        self.img_ref_lq1 = data['img_ref_lq1'].to(self.device)
        self.gt = data['img_in'].to(self.device)  # gt HR

    def optimize_parameters(self, step):

        self.output = self.net_g(self.img_in_lq, self.img_ref_lq1, self.img_ref_lq, self.img_ref)

        if step <= self.net_g_pretrain_steps:
            # pretrain the net_g with pixel Loss
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt)
            l_pix.backward()
            self.optimizer_g.step()
            
            # set log
            self.log_dict['l_g_pix'] = l_pix.item()
        else:
            if self.net_d:
                # train_opt = self.opt['train']
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True
                l2_loss = torch.nn.MSELoss()

                # # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = d_logistic_loss(real_d_pred, fake_d_pred) * 1e-6
                l_d_total = l_d_real + l_d_fake
                self.log_dict['l_d_total'] = l_d_total.item()

                # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_total += d_logistic_loss(real_d_pred, fake_d_pred) 
                real_img_cr_aug = CR_DiffAug(self.gt)
                fake_img_cr_aug = CR_DiffAug(self.output.detach())
                fake_pred_aug = self.net_d(fake_img_cr_aug)
                real_pred_aug = self.net_d(real_img_cr_aug)
                l_d_total += 10 * l2_loss(fake_pred_aug, fake_d_pred) \
                + 10 * l2_loss(real_pred_aug, real_d_pred)

                l_d_total.backward()
                nn.utils.clip_grad_norm_(self.net_d.parameters(), 5.0)
                self.optimizer_d.step()

                if self.cri_grad_penalty and step//16 == 0:
                    self.gt.requires_grad = True

                    real_pred = self.net_d(self.gt)
                    r1_loss = d_r1_loss(real_pred, self.gt)

                    self.net_d.zero_grad()
                    (1* (10 / 2 * r1_loss * 16 + 0 * real_pred[0])).backward()
                    self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
              
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                
                if self.net_d:
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.log_dict['l_g_total'] = l_g_total.item()
                self.optimizer_g.step()
                self.scheduler_g.step()
                # self.scheduler_d.step()
                
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.img_in_lq, self.img_ref_lq1, self.img_ref_lq, self.img_ref)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        out_dict['ref'] = self.img_ref.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['gt_val'] = self.gt.detach().cpu()
            out_dict['ref_val'] = self.img_ref.detach().cpu()
        return out_dict
       
    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        avg_psnr_y = 0.
        avg_ssim_y = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt_val']])
            lq_img, ref_img = tensor2img([visuals['img_in_lq'], visuals['ref_val']])

            if 'padding' in val_data.keys():
                padding = val_data['padding']
                original_size = val_data['original_size']
                if padding:
                    sr_img = sr_img[:original_size[0], :original_size[1]]

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f"{img_name}_{self.opt['name']}.png")
                        
                if self.opt['suffix_lq']:
                    save_img_path_lq = save_img_path.replace(
                            '.png', f'_{self.opt["suffix_lq"]}.png')
                if self.opt['suffix_rec']:
                    save_img_path_rec = save_img_path.replace(
                            '.png', f'_{self.opt["suffix_rec"]}.png')
                if self.opt['suffix_gt']:
                    save_img_path_gt = save_img_path.replace(
                            '.png', f'_{self.opt["suffix_gt"]}.png')
                if self.opt['suffix_ref']:
                    save_img_path_ref = save_img_path.replace(
                            '.png', f'_{self.opt["suffix_ref"]}.png')
                
                mmcv.imwrite(lq_img, save_img_path_lq)
                mmcv.imwrite(sr_img, save_img_path_rec)
                mmcv.imwrite(gt_img, save_img_path_gt)
                mmcv.imwrite(ref_img, save_img_path_ref)

            # tentative for out of GPU memory
            del self.img_in_lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # calculate PSNR
            psnr = metrics.psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'])
            avg_psnr += psnr
            sr_img_y = metrics.bgr2ycbcr(sr_img / 255., only_y=True)
            gt_img_y = metrics.bgr2ycbcr(gt_img / 255., only_y=True)
            psnr_y = metrics.psnr(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_psnr_y += psnr_y
            ssim_y = metrics.ssim(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_ssim_y += ssim_y

            if not self.is_train:
                logger.info(f'# img {img_name} # PSNR: {psnr:.4e} '
                            f'# PSNR_Y: {psnr_y:.4e} # SSIM_Y: {ssim_y:.4e}.')

            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)
        avg_psnr_y = avg_psnr_y / (idx + 1)
        avg_ssim_y = avg_ssim_y / (idx + 1)

        # log
        logger.info(f'# Validation {dataset_name} # PSNR: {avg_psnr:.4e} '
                    f'# PSNR_Y: {avg_psnr_y:.4e} # SSIM_Y: {avg_ssim_y:.4e}.')
        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('psnr_y', avg_psnr_y, current_iter)
            tb_logger.add_scalar('ssim_y', avg_ssim_y, current_iter)
