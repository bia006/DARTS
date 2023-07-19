import cv2
import mmcv
import numpy as np
import torch.utils.data as data

from PIL import Image, ImageOps
from random import randrange

from data.transforms import augment, mod_crop, totensor
from data.util import (paired_paths_from_ann_file,
                            paired_paths_from_folder, paired_paths_from_lmdb)
from utils import FileClient

target_size = (160, 160)
padding_size = (160, 160)

def center_crop(image_name, target_S=(80,80)):

    img_size = image_name.size
    target_h, target_w = target_S[0], target_S[1]

    left = (img_size[0] - target_h)/2
    top = (img_size[1] - target_w)/2
    right = (img_size[0] + target_h)/2
    bottom = (img_size[1] + target_w)/2
    for i in range(1):
      
        area = (left, top, right, bottom)
        c_img = image_name.crop(area)

        fit_img_h = ImageOps.fit(c_img, (target_h, target_w), Image.ANTIALIAS)

        fit_img_l = ImageOps.fit(c_img, (target_h, target_w), Image.ANTIALIAS)

        return fit_img_h


def random_crop(image_name, random_size):

    img_size = image_name.size
    target_h, target_w = random_size[0], random_size[1]

    x_max = img_size[0] - target_h
    y_max = img_size[1] - target_w

    for i in range(1):
        random_x = randrange(0, x_max//2 + 1) * 2
        random_y = randrange(0, y_max//2 + 1) * 2

        area = (random_x, random_y, random_x + target_h, random_y + target_w)
        # area = (left, top, right, bottom)
        c_img = image_name.crop(area)

        fit_img_h = ImageOps.fit(c_img, (target_h, target_w), Image.ANTIALIAS)

        fit_img_l = ImageOps.fit(c_img, (target_h, target_w), Image.ANTIALIAS)

        return fit_img_h


def img_padding(im, right, left, top, bottom):

    width, height = im.size

    new_width = width + right + left
    new_height = height + top + bottom

    result = Image.new(im.mode, (new_width, new_height), (0, 0, 0))

    result.paste(im, (left, top))
    return result


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

        else:
            img_in = mod_crop(img_in, scale)
            img_in_gt = img_in.copy()
            img_ref = mod_crop(img_ref, scale)
            
            
            img_in_h, img_in_w, _ = img_in.shape
            img_ref_h, img_ref_w, _ = img_ref.shape
            padding = False
            if img_in_h != img_ref_h or img_in_w != img_ref_w:
                padding = True
                target_h = max(img_in_h, img_ref_h)
                target_w = max(img_in_w, img_ref_w)
                img_in = mmcv.impad(
                        img_in, shape=(target_h, target_w), pad_val=0)
                img_ref = mmcv.impad(
                        img_ref, shape=(target_h, target_w), pad_val=0)
            
            in_h, in_w, _ = img_in.shape
            ref_h, ref_w, _ = img_ref.shape
            target_h = max(in_h, ref_h)
            target_w = max(in_w, ref_w)
            if target_h != padding_size[0] or target_w != padding_size[0]:
                padding = True
                target_hp = max(target_h, padding_size[0])
                target_wp = max(target_w, padding_size[1])
                img_in = mmcv.impad(
                    img_in, shape=(target_hp, target_wp), pad_val=0)
                img_ref = mmcv.impad(
                    img_ref, shape=(target_hp, target_wp), pad_val=0)
                
            gt_h, gt_w, _ = img_in.shape

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)

        img_ref_lq1 = random_crop(img_ref_pil, random_size=(40,40))
        img_ref_lq = random_crop(img_ref_pil, random_size=(80,80))
        img_ref_g = center_crop(img_ref_pil, (160,160))
        
        # bicubic upsample LR
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.

        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
        img_in_up = img_in_up.astype(np.float32) / 255.

        img_ref_lq = cv2.cvtColor(np.array(img_ref_lq), cv2.COLOR_RGB2BGR)
        img_ref_lq = img_ref_lq.astype(np.float32) / 255.

        img_ref_lq1 = cv2.cvtColor(np.array(img_ref_lq1), cv2.COLOR_RGB2BGR)
        img_ref_lq1 = img_ref_lq1.astype(np.float32) / 255.

        img_ref_g = cv2.cvtColor(np.array(img_ref_g), cv2.COLOR_RGB2BGR)
        img_ref_g = img_ref_g.astype(np.float32) / 255.

        img_ref_up = cv2.cvtColor(np.array(img_ref_up), cv2.COLOR_RGB2BGR)
        img_ref_up = img_ref_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_ref_g, img_ref_lq1, img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up = totensor(  # noqa: E501
            [img_ref_g, img_ref_lq1, img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_ref_g': img_ref_g,
            'img_ref_lq1': img_ref_lq1,
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
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
