import cv2
import mmcv
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from random import randrange

from mmsr.data.transforms import augment, mod_crop, totensor
from mmsr.data.util import (paired_paths_from_ann_file,
                            paired_paths_from_folder, paired_paths_from_lmdb)
from mmsr.utils import FileClient

target_size = 160

def CenterCropping(im, new_width=160, new_height=160):
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im = ImageOps.fit(im, (target_size, target_size), Image.ANTIALIAS)

    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    im = im.astype(np.float32) / 255.

    return im

def center_crop(image_name):

    img_size = image_name.size
    x_max = img_size[0] - target_size
    y_max = img_size[1] - target_size

    left = (img_size[0] - target_size)/2
    top = (img_size[1] - target_size)/2
    right = (img_size[0] + target_size)/2
    bottom = (img_size[1] + target_size)/2
    for i in range(1):
        # random_x = randrange(0, x_max//2 + 1) * 2
        # random_y = randrange(0, y_max//2 + 1) * 2

        # area = (random_x, random_y, random_x + target_size, random_y + target_size)
        area = (left, top, right, bottom)
        c_img = image_name.crop(area)

        fit_img_h = ImageOps.fit(c_img, (target_size, target_size), Image.ANTIALIAS)

        fit_img_l = ImageOps.fit(c_img, (target_size, target_size), Image.ANTIALIAS)

        return fit_img_h


class RefCUFEDDataset(data.Dataset):
    """Reference based CUFED dataset for super-resolution.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'ann_file': Use annotation file to generate paths.
        If opt['io_backend'] != lmdb and opt['ann_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The left.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_in (str): Data root path for input image.
        dataroot_ref (str): Data root path for ref image.
        ann_file (str): Path for annotation file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_flip (bool): Use horizontal and vertical flips.
        use_rot (bool): Use rotation (use transposing h and w for
            implementation).

        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(RefCUFEDDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        if 'filename_tmpl' in opt:  # only used for folder mode
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.in_folder, self.ref_folder]
            self.io_backend_opt['client_keys'] = ['in', 'ref']
            self.paths = paired_paths_from_lmdb(
                [self.in_folder, self.ref_folder], ['in', 'ref'])
        elif 'ann_file' in self.opt:
            self.paths = paired_paths_from_ann_file(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.opt['ann_file'])
        else:
            self.paths = paired_paths_from_folder(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        in_path = self.paths[index]['in_path']
        img_bytes = self.file_client.get(in_path, 'in')
        img_in = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

        if self.opt['phase'] == 'train':
            gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
            # some reference image in CUFED5_train have different sizes
            # resize reference image using PIL bicubic kernel
            img_ref = img_ref * 255
            img_ref = Image.fromarray(
                cv2.cvtColor(img_ref.astype(np.uint8), cv2.COLOR_BGR2RGB))
            img_ref = img_ref.resize((gt_w, gt_h), Image.BICUBIC)
            img_ref = cv2.cvtColor(np.array(img_ref), cv2.COLOR_RGB2BGR)
            img_ref = img_ref.astype(np.float32) / 255.
            # data augmentation
            img_in, img_ref = augment([img_in, img_ref], self.opt['use_flip'],
                                      self.opt['use_rot'])
            img_in_pil = img_in * 255
            img_in_pil = Image.fromarray(
                cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

            img_ref_pil = img_ref * 255
            img_ref_pil = Image.fromarray(
                cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        else:
            # for testing phase, zero padding to image pairs for same size
            img_in_pil = img_in * 255
            img_in_pil = Image.fromarray(
                cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

            img_ref_pil = img_ref * 255
            img_ref_pil = Image.fromarray(
                cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

            img_in_pil = center_crop(img_in_pil)
            img_ref_pil = center_crop(img_ref_pil)

            # img_in = mod_crop(img_in, scale)
            img_in_gt = img_in.copy()
            # img_ref = mod_crop(img_ref, scale)
            img_in_h, img_in_w = img_in_pil.size
            img_ref_h, img_ref_w = img_ref_pil.size
            padding = False

            if img_in_h != img_ref_h or img_in_w != img_ref_w:
                padding = True
                target_h = max(img_in_h, img_ref_h)
                target_w = max(img_in_w, img_ref_w)
                img_in_pil = mmcv.impad(
                    img_in_pil, shape=(target_h, target_w), pad_val=0)
                img_ref_pil = mmcv.impad(
                    img_ref_pil, shape=(target_h, target_w), pad_val=0)

            gt_h, gt_w = img_in_pil.size
        img_gt_val = img_in_pil
        img_ref_val = img_ref_pil
        img_gt_val = cv2.cvtColor(np.array(img_gt_val), cv2.COLOR_RGB2BGR)
        img_gt_val = img_gt_val.astype(np.float32) / 255.

        img_ref_val = cv2.cvtColor(np.array(img_ref_val), cv2.COLOR_RGB2BGR)
        img_ref_val = img_ref_val.astype(np.float32) / 255.

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        # center crop images, ref and input images

        img_in_8x = CenterCropping(img_in_pil, 20, 20)
        img_ref_8x = CenterCropping(img_ref_pil, 20, 20)
        img_in_4x = CenterCropping(img_in_pil, 40, 40)
        img_ref_4x = CenterCropping(img_ref_pil, 40, 40)
        img_in_2x = CenterCropping(img_in_pil, 80, 80)
        img_ref_2x = CenterCropping(img_ref_pil, 80, 80)          

        # bicubic upsample LR
        # img_in_8x = img_in_8x.resize((20, 20), Image.BICUBIC)
        # img_ref_8x = img_ref_8x.resize((20, 20), Image.BICUBIC)

        # img_in_4x = img_in_4x.resize((40, 40), Image.BICUBIC)
        # img_ref_4x = img_ref_4x.resize((40, 40), Image.BICUBIC)

        # img_in_2x = img_in_2x.resize((80, 80), Image.BICUBIC)
        # img_ref_2x = img_ref_2x.resize((80, 80), Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.

        img_ref_lq = cv2.cvtColor(np.array(img_ref_lq), cv2.COLOR_RGB2BGR)
        img_ref_lq = img_ref_lq.astype(np.float32) / 255.
        
        img_in_4x = cv2.cvtColor(np.array(img_in_4x), cv2.COLOR_RGB2BGR)
        img_in_4x = img_in_4x.astype(np.float32) / 255.
        img_in_2x = cv2.cvtColor(np.array(img_in_2x), cv2.COLOR_RGB2BGR)
        img_in_2x = img_in_2x.astype(np.float32) / 255.

        img_ref_4x = cv2.cvtColor(np.array(img_ref_4x), cv2.COLOR_RGB2BGR)
        img_ref_4x = img_ref_4x.astype(np.float32) / 255.
        img_ref_2x = cv2.cvtColor(np.array(img_ref_2x), cv2.COLOR_RGB2BGR)
        img_ref_2x = img_ref_2x.astype(np.float32) / 255.
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_val, img_ref_val, img_in, img_in_lq, img_ref, img_ref_lq, img_in_8x, img_in_4x, img_in_2x, img_ref_8x, img_ref_4x, img_ref_2x = totensor(  # noqa: E501
            [img_gt_val, img_ref_val, img_in, img_in_lq, img_ref, img_ref_lq, img_in_8x, img_in_4x, img_in_2x, img_ref_8x, img_ref_4x, img_ref_2x],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_gt_val': img_gt_val,
            'img_ref_val': img_ref_val,
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            # 'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            # 'img_ref_up': img_ref_up,
            'img_in_8x': img_in_8x,
            'img_in_4x': img_in_4x,
            'img_in_2x': img_in_2x,
            'img_ref_8x': img_ref_8x,
            'img_ref_4x': img_ref_4x,
            'img_ref_2x': img_ref_2x,
        }

        if self.opt['phase'] != 'train':
            img_in_gt = totensor(img_in_gt, bgr2rgb=True, float32=True)
            return_dict['img_in'] = img_in_gt
            return_dict['lq_path'] = ref_path
            return_dict['padding'] = padding
            return_dict['original_size'] = (img_in_h, img_in_w)

        return return_dict

    def __len__(self):
        return len(self.paths)
