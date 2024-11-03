import cv2
import math
import numpy as np
import random
import torch
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop

AUGMENT_OPT = {
    'use_hflip': False,
    'use_rot': False
}

KERNEL_OPT = {
    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4],
    'betap_range': [1, 2],

    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    'final_sinc_prob': 0.8,
}

DEGRADE_OPT = {
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.15, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 30],
    'poisson_scale_range': [0.05, 3],
    'gray_noise_prob': 0.4,
    'jpeg_range': [30, 95],

    # the second degradation process
    'second_blur_prob': 0.8,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 25],
    'poisson_scale_range2': [0.05, 2.5],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [30, 95],

    'gt_size': 512,
    'no_degradation_prob': 0.01,
    'use_usm': True,
    'sf': 4,
    'random_size': False,
    'resize_lq': True
}

class RealESRGANDegradation:

    def __init__(self, augment_opt=None, kernel_opt=None, degrade_opt=None, device='cuda', resolution=None):
        if augment_opt is None:
            augment_opt = AUGMENT_OPT
        self.augment_opt = augment_opt
        if kernel_opt is None:
            kernel_opt = KERNEL_OPT
        self.kernel_opt = kernel_opt
        if degrade_opt is None:
            degrade_opt = DEGRADE_OPT
        self.degrade_opt = degrade_opt
        if resolution is not None:
            self.degrade_opt['gt_size'] = resolution
        self.device = device

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = USMSharp().to(self.device)

        # blur settings for the first degradation
        self.blur_kernel_size = kernel_opt['blur_kernel_size']
        self.kernel_list = kernel_opt['kernel_list']
        self.kernel_prob = kernel_opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = kernel_opt['blur_sigma']
        self.betag_range = kernel_opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = kernel_opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = kernel_opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = kernel_opt['blur_kernel_size2']
        self.kernel_list2 = kernel_opt['kernel_list2']
        self.kernel_prob2 = kernel_opt['kernel_prob2']
        self.blur_sigma2 = kernel_opt['blur_sigma2']
        self.betag_range2 = kernel_opt['betag_range2']
        self.betap_range2 = kernel_opt['betap_range2']
        self.sinc_prob2 = kernel_opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = kernel_opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def get_kernel(self):

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.kernel_opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.kernel_opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.kernel_opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return (kernel, kernel2, sinc_kernel)

    @torch.no_grad()
    def __call__(self, img_gt, kernels=None):
        '''
            :param: img_gt: BCHW, RGB, [0, 1] float32 tensor
        '''
        if kernels is None:
            kernel = []
            kernel2 = []
            sinc_kernel = []
            for _ in range(img_gt.shape[0]):
                k, k2, sk = self.get_kernel()
                kernel.append(k)
                kernel2.append(k2)
                sinc_kernel.append(sk)
            kernel = torch.stack(kernel)
            kernel2 = torch.stack(kernel2)
            sinc_kernel = torch.stack(sinc_kernel)
        else:
            # kernels created in dataset.
            kernel, kernel2, sinc_kernel = kernels

        # ----------------------- Pre-process ----------------------- #
        im_gt = img_gt.to(self.device)
        if self.degrade_opt['use_usm']:
            im_gt = self.usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel = kernel.to(self.device)
        kernel2 = kernel2.to(self.device)
        sinc_kernel = sinc_kernel.to(self.device)
        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degrade_opt['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.degrade_opt['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degrade_opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = torch.nn.functional.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.degrade_opt['gray_noise_prob']
        if random.random() < self.degrade_opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degrade_opt['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.degrade_opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degrade_opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.degrade_opt['second_blur_prob']:
            out = out.contiguous()
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degrade_opt['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.degrade_opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degrade_opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = torch.nn.functional.interpolate(
                out,
                size=(int(ori_h / self.degrade_opt['sf'] * scale),
                      int(ori_w / self.degrade_opt['sf'] * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.degrade_opt['gray_noise_prob2']
        if random.random() < self.degrade_opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degrade_opt['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.degrade_opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = torch.nn.functional.interpolate(
                    out,
                    size=(ori_h // self.degrade_opt['sf'],
                          ori_w // self.degrade_opt['sf']),
                    mode=mode,
                    )
            out = out.contiguous()
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degrade_opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degrade_opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = torch.nn.functional.interpolate(
                    out,
                    size=(ori_h // self.degrade_opt['sf'],
                          ori_w // self.degrade_opt['sf']),
                    mode=mode,
                    )
            out = out.contiguous()
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.degrade_opt['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.degrade_opt['sf'])

        if self.degrade_opt['resize_lq']:
            im_lq = torch.nn.functional.interpolate(
                    im_lq,
                    size=(im_gt.size(-2),
                          im_gt.size(-1)),
                    mode='bicubic',
                    )

        if random.random() < self.degrade_opt['no_degradation_prob'] or torch.isnan(im_lq).any():
            im_lq = im_gt

        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        im_lq = im_lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        im_lq = im_lq*2 - 1.0
        im_gt = im_gt*2 - 1.0

        if self.degrade_opt['random_size']:
            raise NotImplementedError
            im_lq, im_gt = self.randn_cropinput(im_lq, im_gt)

        im_lq = torch.clamp(im_lq, -1.0, 1.0)
        im_gt = torch.clamp(im_gt, -1.0, 1.0)

        return (im_lq, im_gt)