import argparse
import logging
import math
import os.path as osp
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import os
# import tensorflow as tf
import json
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms.functional as F
from mmcv.runner import get_time_str, init_dist
from torch import distributed as dist
from mmsr.mmsr.models.archs import ref_restoration_arch as vits

from mmsr.data import create_dataloader, create_dataset
from mmsr.data.data_sampler import DistIterSampler
from mmsr.models import create_model
from mmsr.utils import (MessageLogger, get_root_logger, init_tb_logger,
                        make_exp_dirs, set_random_seed)
from mmsr.utils.options import dict2str, dict_to_nonedict, parse
# from mmsr.utils.util import check_resume
# from mmsr.utils import ProgressBar, tensor2img
import utils_dino

try:
    import wandb
except ImportError:
    wandb = None

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

def feed_data(self, data):
    self.img_in_lq = data['img_in_lq'].to(self.device) # LR/ gt_down
    self.img_in_up = data['img_in_up'].to(self.device) # LR_bicubic_up
    self.img_ref = data['img_ref'].to(self.device) # ref HR
    self.img_ref_lq = data['img_ref_lq'].to(self.device) # ref HR downsampled
    self.gt = data['img_in'].to(self.device)  # gt HR
    self.match_img_in = data['img_in_up'].to(self.device)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# class Visualizer():
#     def __init__(self, args):
#         self.args = args
#         self.tf = tf
#         self.log_dir = os.path.join(args.checkpoint_path, 'logs')
#         self.writer = tf.summary.FileWriter(self.log_dir)
    
#     def plot_loss(self, loss, step, tag):
#         summary = self.tf.Summary(
#             value=[self.tf.Summary.Value(tag=tag, simple_value=loss)])
#         self.writer.add_summary(summary, step)

#     def plot_dict(self, loss, step):
#         for tag, value in loss.items():
#             summary = self.tf.Summary(
#                 value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
#             self.writer.add_summary(summary, step)


def get_args_parser():
    # options
    parser = argparse.ArgumentParser('RefSR_TR', add_help=False)
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_false", help='Whether to use wandb record training')
    parser.add_argument("--wandb_project_name", type=str, default='SR_swinTR', help='Project name')
    parser.add_argument('--tf_log', action="store_true", help='If we use tensorboard file')
    parser.add_argument('--checkpoint_path', default='/tmp', type=str, help='Save checkpoints')

    parser.add_argument('--arch', default='SwinTransformer', type=str,
        help="""Name of architecture to train.""")
    parser.add_argument('--patch_size', default=2, type=int, help="""Size in pixels of input square patches.""")
    parser.add_argument('--img_size', default=40, type=int, help="""Size in input images for the transformer encoder/decoder.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization Params
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str, help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_RefSR_TR(args):
    utils_dino.init_distributed_mode(args)
    utils_dino.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils_dino.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    opt = parse(args.opt, is_train=True)

    if wandb and args.use_wandb:
        wandb.init(project='SR_Swin_TR', name=args.wandb_project_name, reinit=True)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = n_gpu > 1

    # # distributed training settings
    # if args.launcher == 'none':  # disabled distributed training
    #     opt['dist'] = False
    #     rank = -1
    #     print('Disabled distributed training.', flush=True)
    # else:
    #     opt['dist'] = True
    #     if args.launcher == 'slurm' and 'dist_params' in opt:
    #         init_dist(args.launcher, **opt['dist_params'])
    #     else:
    #         init_dist(args.launcher)
    #     world_size = torch.distributed.get_world_size()
    #     rank = torch.distributed.get_rank()

    # # load resume states if exists
    # if opt['path'].get('resume_state', None):
    #     device_id = torch.cuda.current_device()
    #     resume_state = torch.load(
    #         opt['path']['resume_state'],
    #         map_location=lambda storage, loc: storage.cuda(device_id))
    #     check_resume(opt, resume_state['iter'])
    # else:
    #     resume_state = None

    # # mkdir and loggers
    # if resume_state is None:
    #     make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # # initialize tensorboard logger
    # tb_logger = None
    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #     tb_logger = init_tb_logger(log_dir='./tb_logger/' + opt['name'])

    # # convert to NoneDict, which returns None for missing keys
    # opt = dict_to_nonedict(opt)

    # # random seed
    # seed = opt['train']['manual_seed']
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # logger.info(f'Random seed: {seed}')
    # set_random_seed(seed)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloaders
    # ToDo: add augmentation to training 
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # dataset_ratio: enlarge the size of datasets for each epoch
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            # if opt['dist']:
            train_sampler = DistIterSampler(train_set, utils_dino.get_world_size(), torch.distributed.get_rank(),
                                                dataset_enlarge_ratio)
            total_epochs = total_iters / (
                    train_size * dataset_enlarge_ratio)
            total_epochs = int(math.ceil(total_epochs))
            # else:
                # train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt,
                                             train_sampler)
            logger.info(
                f'Number of train images: {len(train_set)}, iters: {train_size}'
            )
            logger.info(
                f'Total epochs needed: {total_epochs} for iters {total_iters}')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info(
                f"Number of val images/folders in {dataset_opt['name']}: "
                f'{len(val_set)}')
        else:
            raise NotImplementedError(f'Phase {phase} is not recognized.')
    assert train_loader is not None

    # create model
    student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            img_size=args.img_size,  # stochastic depth
        )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size//2,
                    img_size=args.img_size*4,)
    embed_dim = student.embed_dim
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils_dino.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    # teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

     # ============ preparing loss ... ============
    # dino_loss = DINOLoss(
    #     args.out_dim,
    #     args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
    #     args.warmup_teacher_temp,
    #     args.teacher_temp,
    #     args.warmup_teacher_temp_epochs,
    #     args.epochs,
    # ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils_dino.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils_dino.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    # if args.use_fp16:
    #     fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils_dino.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils_dino.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils_dino.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_dino.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(train_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils_dino.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        # fp16_scaler=fp16_scaler,
        # dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        print ('sampler', train_sampler)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils_dino.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils_dino.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils_dino.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils_dino.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images_list = [
        ref_lr: feed_data(images['img_ref_lq']), 
        ref: feed_data(images['img_ref']), 
        lr: feed_data(images['img_in_lq']),
        lr_up: feed_data(images['img_in_up']),
        gt: feed_data(images['img_in'])]

        images = [im.cuda(non_blocking=True) for im in images_list]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils_dino.clip_gradients(student, args.clip_grad)
            utils_dino.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils_dino.clip_gradients(student, args.clip_grad)
            utils_dino.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}









    # # resume training
    # if resume_state:
    #     logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
    #                 f"iter: {resume_state['iter']}.")
    #     start_epoch = resume_state['epoch']
    #     current_iter = resume_state['iter']
    #     student.resume_training(resume_state)  # handle optimizers and schedulers
    # else:
    #     current_iter = 0
    #     start_epoch = 0

    # # create message logger (formatted outputs)
    # msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # # training
    # logger.info(
    #     f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    # data_time, iter_time = 0, 0












    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            student.update_learning_rate(
                current_iter, warmup_iter=opt['train']['warmup_iter'])
            # training
            student.feed_data(train_data)
            student.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if get_rank() == 0:
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': student.get_current_learning_rate()})
                    log_vars.update({'time': iter_time, 'data_time': data_time})
                    log_vars.update(student.get_current_log())
                    msg_logger(log_vars)

                    visuals = student.get_current_visuals()
                    losses = student.get_current_log()
                    print ('losses', losses)
                    
                    in_img, sr_img, gt_img = visuals['img_in_lq'], visuals['reconstructed'], visuals['gt']
                    vis_img = [in_img, sr_img, gt_img]

                
                    vis_loss = {
                    # 'd_loss': d_loss_val,
                    # 'g_loss': g_loss_val,
                    'pixel_loss': losses['l_pix'],
                    }
                    wandb.log(vis_loss, step=current_iter)
                    wandb.log({"images": [wandb.Image(image) for image in vis_img]})
                
                    if args.tf_log:
                        Visualizer.plot_dict(vis_loss, step=(epoch * args.batch * int(os.environ["WORLD_SIZE"])))


            # validation
            if opt['datasets'][
                    'val'] and current_iter % opt['val']['val_freq'] == 0:
                student.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                student.save(epoch, current_iter)

            data_time = time.time()
            iter_time = time.time()
        # end of iter
    # end of epoch

    logger.info('End of training.')
    logger.info('Saving the latest model.')
    student.save(epoch=-1, current_iter=-1)  # -1 for the latest

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RefSR_TR', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_RefSR_TR(args)
