import importlib
import numpy as np
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
import torch.nn as nn

import mmsr.models.networks as networks
import mmsr.utils.metrics as metrics
from mmsr.utils import ProgressBar, tensor2img

from .sr_model import SRModel

loss_module = importlib.import_module('mmsr.models.losses')
logger = logging.getLogger('base')

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class SwinTransformerModel(SRModel):

    def __init__(self, opt):
        super(SwinTransformerModel, self).__init__(opt)

        load_path = self.opt['path'].get('pretrain_model_student', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
            self.load_network(self.net_teacher, load_path,
                              self.opt['path']['strict_load'])
        if self.is_train:
            self.net_student.train()
            # self.net_teacher.train()
            print(f"Student and Teacher are built: they are both Transformers network. ")

            # optimizers
            train_opt = self.opt['train']
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
            optim_params_offset = []
            optim_params_relu2_offset = []
            optim_params_relu3_offset = []
            if train_opt.get('lr_relu3_offset', None):
                optim_params_relu3_offset = []
            for name, v in self.net_student.named_parameters():
                if v.requires_grad:
                    # if 'offset' in name:
                    #     if 'small' in name:
                    #         logger.info(name)
                    #         optim_params_relu3_offset.append(v)
                    #     elif 'medium' in name:
                    #         logger.info(name)
                    #         optim_params_relu2_offset.append(v)
                    #     else:
                    #         optim_params_offset.append(v)
                    # else:
                    optim_params_g.append(v)

            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g
                # }, {
                #     'params': optim_params_offset,
                #     'lr': train_opt['lr_offset']
                # }, {
                #     'params': optim_params_relu3_offset,
                #     'lr': train_opt['lr_relu3_offset']
                # }, {
                #     'params': optim_params_relu2_offset,
                #     'lr': train_opt['lr_relu2_offset']
                }],
                lr=train_opt['lr_g'],
                weight_decay=weight_decay_g,
                betas=train_opt['beta_g'])

            self.optimizers.append(self.optimizer_g)
            # for p in self.net_teacher.parameters():
            #     p.requires_grad = False

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

        # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        # if train_opt.get('dino_opt'):
        #     cri_dino_cls = getattr(loss_module, 'DINOLoss')
        #     self.cri_dino = cri_dino_cls(
        #         **train_opt['dino_opt']).to(self.device)
        # else:
        #     logger.info('Remove dino loss.')

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
            cri_texture_cls = getattr(loss_module, 'TextureLoss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('maskedFM_opt', None):
            cri_maskedFM_cls = getattr(loss_module, 'MaskedFMLoss')
            self.cri_maskedFM = cri_maskedFM_cls(**train_opt['maskedFM_opt']).to(
                self.device)
        else:
            logger.info('Remove maskedFM loss.')
            self.cri_maskedFM = None

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
        self.net_student_pretrain_steps = train_opt['net_student_pretrain_steps']
        # self.net_teacher_pretrain_steps = train_opt['net_teacher_pretrain_steps']
        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        # optimizers
        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        # check the schedulers
        self.setup_schedulers()
        print(f"The discriminator is built: it is a styleGAN discriminator.")

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device) # LR/ gt_down
        self.img_in_up = data['img_in_up'].to(self.device) # LR_bicubic_up
        self.img_ref = data['img_ref'].to(self.device) # ref HR
        self.img_ref_lq = data['img_ref_lq'].to(self.device) # ref HR downsampled
        self.gt = data['img_in'].to(self.device)  # gt HR
        self.match_img_in = data['img_in_up'].to(self.device)

        self.img_in_8x = data['img_in_8x'].to(self.device)
        self.img_in_4x = data['img_in_4x'].to(self.device) 
        self.img_in_2x = data['img_in_2x'].to(self.device) 
        self.img_ref_8x = data['img_ref_8x'].to(self.device)
        self.img_ref_4x = data['img_ref_4x'].to(self.device) 
        self.img_ref_2x = data['img_in_2x'].to(self.device) 

        self.img_in_t = [self.img_in_8x, self.img_in_4x, self.img_in_2x, self.gt]
        self.img_ref_t = [self.img_ref_8x, self.img_ref_4x, self.img_ref_2x, self.img_ref]
        
        
    def optimize_parameters(self, step):
        self.output = self.net_student(self.img_in_lq, self.img_ref_lq)
        # self.teacher_feat = self.net_teacher(self.img_in_t, self.img_ref_t)
        

        if step <= self.net_student_pretrain_steps:
            # pretrain the net_g with pixel Loss
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt) 
            # l_pix_t = self.cri_pix(self.teacher_feat, self.gt) 
            self.optimizer_g.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
            # self.log_dict['l_pix_t'] = l_pix_t.item()
        else:
            if self.net_d:
                # train net_d
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True
                # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = l_d_real + l_d_fake
                if self.cri_grad_penalty:
                    l_grad_penalty = self.cri_grad_penalty(
                        self.net_d, self.gt, self.output)
                    self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
                l_d_total += l_grad_penalty
                self.log_dict['l_d_total'] = l_d_total.item()
                l_d_total.backward()
                self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if (step - self.net_student_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_student_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                    # l_g_pix_t = self.cri_pix(self.teacher_feat, self.gt)
                    # l_g_total += l_g_pix_t
                    # self.log_dict['l_g_pix_t'] = l_g_pix_t.item()
                # if self.cri_dino:
                #     l_dino = self.cri_dino(self.dino_s, self.dino_t, step)
                #     l_g_total += l_dino
                #     self.log_dict['l_dino'] = l_dino.item()
                # if self.cri_perceptual:
                #     l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                #     l_g_total += l_g_percep
                #     self.log_dict['l_g_percep'] = l_g_percep.item()
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                if self.cri_texture:
                    l_g_texture = self.cri_texture(self.output, self.maps,
                                                   self.weights)
                    l_g_total += l_g_texture
                    self.log_dict['l_g_texture'] = l_g_texture.item()
                if self.cri_maskedFM:
                    l_g_maskedFM = self.cri_maskedFM(self.output, self.gt)
                    l_g_total += l_g_maskedFM


                if self.net_d:
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.optimizer_g.step()
                # momentum_schedule = cosine_scheduler(self.opt['train']['momentum_teacher'], 1, step, self.opt['train']['niter'])
                # EMA update for the teacher
                # with torch.no_grad():
                #     m = momentum_schedule[step]  # momentum parameter
                #     for param_q, param_k in zip(self.net_student.parameters(), self.net_teacher.parameters()):
                #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def test(self):
        self.net_student.eval()
        with torch.no_grad():
            self.output, self.ref = self.net_student(self.img_in_lq, self.img_ref_lq)
        self.net_student.train()
        # self.net_teacher.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['reconstructed'] = self.output.detach().cpu()
        out_dict['reference'] = self.img_ref.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_student, 'net_s', current_iter)
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
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt']])

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
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.png', f'_{self.opt["suffix"]}.png')
                mmcv.imwrite(sr_img, save_img_path)

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
