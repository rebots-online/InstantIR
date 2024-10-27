#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The LCM team and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import functools
import gc
import logging
import pyrallis
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from collections import namedtuple
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    AutoImageProcessor, AutoModel
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, resolve_interpolation_mode
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from basicsr.utils.degradation_pipeline import RealESRGANDegradation
from utils.train_utils import (
    seperate_ip_params_from_unet,
    import_model_class_from_model_name_or_path,
    tensor_to_pil,
    get_train_dataset, prepare_train_dataset, collate_fn,
    encode_prompt, importance_sampling_fn, extract_into_tensor

)
from data.data_config import DataConfig
from losses.loss_config import LossesConfig
from losses.losses import *

from module.ip_adapter.resampler import Resampler
from module.ip_adapter.utils import init_adapter_in_unet, prepare_training_image_embeds


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

logger = get_logger(__name__)


def prepare_latents(lq, vae, scheduler, generator, timestep):
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])
    lq_pt = [transform(lq_pil.convert("RGB")) for lq_pil in lq]
    img_pt = torch.stack(lq_pt).to(vae.device, dtype=vae.dtype)
    img_pt = img_pt * 2.0 - 1.0
    with torch.no_grad():
        latents = vae.encode(img_pt).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    noise = torch.randn(latents.shape, generator=generator, device=vae.device, dtype=vae.dtype, layout=torch.strided).to(vae.device)
    bsz = latents.shape[0]
    print(f"init latent at {timestep}")
    timestep = torch.tensor([timestep]*bsz, device=vae.device, dtype=torch.int64)
    latents = scheduler.add_noise(latents, noise, timestep)
    return latents


def log_validation(unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                   scheduler, image_encoder, image_processor,
                   args, accelerator, weight_dtype, step, lq_img=None, gt_img=None, is_final_validation=False, log_local=False):
    logger.info("Running validation... ")

    image_logs = []

    lq = [Image.open(lq_example) for lq_example in args.validation_image]

    pipe = StableDiffusionXLPipeline(
            vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            unet, scheduler, image_encoder, image_processor,
        ).to(accelerator.device)

    timesteps = [args.num_train_timesteps - 1]
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    latents = prepare_latents(lq, vae, scheduler, generator, timesteps[-1])
    image = pipe(
        prompt=[""]*len(lq),
        ip_adapter_image=[lq],
        num_inference_steps=1,
        timesteps=timesteps,
        generator=generator,
        guidance_scale=1.0,
        height=args.resolution,
        width=args.resolution,
        latents=latents,
    ).images

    if log_local:
        # for i, img in enumerate(tensor_to_pil(lq_img)):
        #     img.save(f"./lq_{i}.png")
        # for i, img in enumerate(tensor_to_pil(gt_img)):
        #     img.save(f"./gt_{i}.png")
        for i, img in enumerate(image):
            img.save(f"./lq_IPA_{i}.png")
        return

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = [np.asarray(pil_img) for pil_img in image]
            images = np.stack(images, axis=0)
            if lq_img is not None and gt_img is not None:
                input_lq = lq_img.detach().cpu()
                input_lq = np.asarray(input_lq.add(1).div(2).clamp(0, 1))
                input_gt = gt_img.detach().cpu()
                input_gt = np.asarray(input_gt.add(1).div(2).clamp(0, 1))
                tracker.writer.add_images("lq", input_lq, step, dataformats="NCHW")
                tracker.writer.add_images("gt", input_gt, step, dataformats="NCHW")
            tracker.writer.add_images("rec", images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            raise NotImplementedError("Wandb logging not implemented for validation.")
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps

        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lcm_lora_path",
        type=str,
        default=None,
        help="Path to LCM lora or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--feature_extractor_path",
        type=str,
        default=None,
        help="Path to image encoder for IP-Adapters or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_adapter_model_path",
        type=str,
        default=None,
        help="Path to IP-Adapter models or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_tokens",
        type=int,
        default=64,
        help="Number of tokens to use in IP-adapter cross attention mechanism.",
    )
    parser.add_argument(
        "--use_clip_encoder",
        action="store_true",
        help="Whether or not to use DINO as image encoder, else CLIP encoder.",
    )
    parser.add_argument(
        "--image_encoder_hidden_feature",
        action="store_true",
        help="Whether or not to use the penultimate hidden states as image embeddings.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=4000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--save_only_adapter",
        action="store_true",
        help="Only save extra adapter to save space.",
    )
    # ----Image Processing----
    parser.add_argument(
        "--data_config_path",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--text_drop_rate",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--image_drop_rate",
        type=float,
        default=0,
        help="Proportion of IP-Adapter inputs to be dropped. Defaults to 0 (no drop-out).",
    )
    parser.add_argument(
        "--cond_drop_rate",
        type=float,
        default=0,
        help="Proportion of all conditions to be dropped. Defaults to 0 (no drop-out).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Diffusion Training Arguments----
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=3.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--losses_config_path",
        type=str,
        default='config_files/losses.yaml',
        required=True,
        help=("A yaml file containing losses to use and their weights."),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help=(
            "The value of the LoRA alpha parameter, which controls the scaling factor in front of the LoRA weight"
            " update delta_W. No scaling will be performed if this value is equal to `lora_rank`."
        ),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "A comma-separated string of target module keys to add LoRA to. If not set, a default list of modules will"
            " be used. By default, LoRA will be applied to all conv and linear layers."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        required=False,
        help=(
            "The batch size used when encoding (and decoding) images to latents (and vice versa) using the VAE."
            " Encoding or decoding the whole batch at once may run into OOM issues."
        ),
    )
    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=3000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help=(
            "sanity check"
        ),
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="trian",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.teacher_revision
    )
    noise_scheduler.config.num_train_timesteps = args.num_train_timesteps
    lcm_scheduler = LCMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SDXL checkpoint.
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.teacher_revision, use_fast=False
    )

    # 3. Load text encoders from SDXL checkpoint.
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.teacher_revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.teacher_revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.teacher_revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.teacher_revision
    )

    if args.use_clip_encoder:
        image_processor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.feature_extractor_path)
    else:
        image_processor = AutoImageProcessor.from_pretrained(args.feature_extractor_path)
        image_encoder = AutoModel.from_pretrained(args.feature_extractor_path)

    # 4. Load VAE from SDXL checkpoint (or more stable VAE)
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.teacher_revision,
    )

    # 7. Create online student U-Net.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.teacher_revision
    )

    # Resampler for project model in IP-Adapter
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.adapter_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )

    # Load the same adapter in both unet.
    init_adapter_in_unet(
        unet,
        image_proj_model,
        os.path.join(args.pretrained_adapter_model_path, 'adapter_ckpt.pt'),
        adapter_tokens=args.adapter_tokens,
    )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.pretrained_lcm_lora_path is not None:
        lora_state_dict, alpha_dict = StableDiffusionXLPipeline.lora_state_dict(args.pretrained_lcm_lora_path)
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        lora_state_dict = dict()
        for k, v in unet_state_dict.items():
            if "ip" in k:
                k = k.replace("attn2", "attn2.processor")
                lora_state_dict[k] = v
            else:
                lora_state_dict[k] = v
        if alpha_dict:
            args.lora_alpha = next(iter(alpha_dict.values()))
        else:
            args.lora_alpha = 1
    # 9. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.lora_target_modules is not None:
        lora_target_modules = [module_key.strip() for module_key in args.lora_target_modules.split(",")]
    else:
        lora_target_modules = [
            "to_q",
            "to_kv",
            "0.to_out",
            "attn1.to_k",
            "attn1.to_v",
            "to_k_ip",
            "to_v_ip",
            "ln_k_ip.linear",
            "ln_v_ip.linear",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Legacy
    # for k, v in lcm_pipe.unet.state_dict().items():
    #     if "lora" in k or "base_layer" in k:
    #         lcm_dict[k.replace("default_0", "default")] = v

    unet.add_adapter(lora_config)
    if args.pretrained_lcm_lora_path is not None:
        incompatible_keys = set_peft_model_state_dict(unet, lora_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    # 6. Freeze unet, vae, text_encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if args.save_only_adapter:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if isinstance(model, type(unwrap_model(unet))):  # save adapter only
                        unet_ = unwrap_model(model)
                        # also save the checkpoints in native `diffusers` format so that it can be easily
                        # be independently loaded via `load_lora_weights()`.
                        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_))
                        StableDiffusionXLPipeline.save_lora_weights(output_dir, unet_lora_layers=state_dict, safe_serialization=False)

                    weights.pop()

        def load_model_hook(models, input_dir):

            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(unwrap_model(unet))):
                    unet_ = unwrap_model(model)
                    lora_state_dict, _ = StableDiffusionXLPipeline.lora_state_dict(input_dir)
                    unet_state_dict = {
                        f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
                    }
                    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
                    lora_state_dict = dict()
                    for k, v in unet_state_dict.items():
                        if "ip" in k:
                            k = k.replace("attn2", "attn2.processor")
                            lora_state_dict[k] = v
                        else:
                            lora_state_dict[k] = v
                    incompatible_keys = set_peft_model_state_dict(unet_, lora_state_dict, adapter_name="default")
                    if incompatible_keys is not None:
                        # check only for unexpected keys
                        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                        if unexpected_keys:
                            logger.warning(
                                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                                f" {unexpected_keys}. "
                            )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    lora_params, non_lora_params = seperate_lora_params_from_unet(unet)
    params_to_optimize = lora_params
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    datasets = []
    datasets_name = []
    datasets_weights = []
    deg_pipeline = RealESRGANDegradation(device=accelerator.device, resolution=args.resolution)
    if args.data_config_path is not None:
        data_config: DataConfig = pyrallis.load(DataConfig, open(args.data_config_path, "r"))
        for single_dataset in data_config.datasets:
            datasets_weights.append(single_dataset.dataset_weight)
            datasets_name.append(single_dataset.dataset_folder)
            dataset_dir = os.path.join(args.train_data_dir, single_dataset.dataset_folder)
            image_dataset = get_train_dataset(dataset_dir, dataset_dir, args, accelerator)
            image_dataset = prepare_train_dataset(image_dataset, accelerator, deg_pipeline)
            datasets.append(image_dataset)
        # TODO: Validation dataset
        if data_config.val_dataset is not None:
            val_dataset = get_train_dataset(dataset_name, dataset_dir, args, accelerator)
    logger.info(f"Datasets mixing: {list(zip(datasets_name, datasets_weights))}")

    # Mix training datasets.
    sampler_train = None
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        # Weighted each dataset
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        dataset_weights = []
        for single_dataset, single_weight in zip(datasets, datasets_weights):
            dataset_weights.extend([len(train_dataset) / len(single_dataset) * single_weight] * len(single_dataset))
        sampler_train = torch.utils.data.WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=len(dataset_weights)
        )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        shuffle=True if sampler_train is None else False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 14. Embeddings for the UNet.
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    def compute_embeddings(prompt_batch, original_sizes, crop_coords, text_encoders, tokenizers, is_train=True):
        def compute_time_ids(original_size, crops_coords_top_left):
            target_size = (args.resolution, args.resolution)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
            return add_time_ids

        prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch, text_encoders, tokenizers, is_train)
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = torch.cat([compute_time_ids(s, c) for s, c in zip(original_sizes, crop_coords)])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    compute_embeddings_fn = functools.partial(compute_embeddings, text_encoders=text_encoders, tokenizers=tokenizers)

    # Move pixels into latents.
    @torch.no_grad()
    def convert_to_latent(pixels):
        model_input = vae.encode(pixels).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        if args.pretrained_vae_model_name_or_path is None:
            model_input = model_input.to(weight_dtype)
        return model_input

    # 15. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # 16. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 8. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    for p in non_lora_params:
        p.data = p.data.to(dtype=weight_dtype)
    for p in lora_params:
        p.requires_grad_(True)
    unet.to(accelerator.device)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # Instantiate Loss.
    losses_configs: LossesConfig = pyrallis.load(LossesConfig, open(args.losses_config_path, "r"))
    lcm_losses = list()
    for loss_config in losses_configs.lcm_losses:
        logger.info(f"Loading lcm loss: {loss_config.name}")
        loss = namedtuple("loss", ["loss", "weight"])
        loss_class = eval(loss_config.name)
        lcm_losses.append(loss(loss_class(
            visualize_every_k=loss_config.visualize_every_k, 
            dtype=weight_dtype,
            accelerator=accelerator,
            dino_model=image_encoder,
            dino_preprocess=image_processor,
            huber_c=args.huber_c,
            **loss_config.init_params), weight=loss_config.weight))

    # Final check.
    for n, p in unet.named_parameters():
        if p.requires_grad:
            assert "lora" in n, n
            assert p.dtype == torch.float32, n
        else:
            assert "lora" not in n, f"{n}"
            assert p.dtype == weight_dtype, n
    if args.sanity_check:
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))

        # Check input data
        batch = next(iter(train_dataloader))
        lq_img, gt_img = deg_pipeline(batch["images"], (batch["kernel"], batch["kernel2"], batch["sinc_kernel"]))
        out_images = log_validation(unwrap_model(unet), vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two,
                            lcm_scheduler, image_encoder, image_processor,
                            args, accelerator, weight_dtype, step=0, lq_img=lq_img, gt_img=gt_img, is_final_validation=False, log_local=True)
        exit()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # 17. Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                total_loss = torch.tensor(0.0)
                bsz = batch["images"].shape[0]

                # Drop conditions.
                rand_tensor = torch.rand(bsz)
                drop_image_idx = rand_tensor < args.image_drop_rate
                drop_text_idx = (rand_tensor >= args.image_drop_rate) & (rand_tensor < args.image_drop_rate + args.text_drop_rate)
                drop_both_idx = (rand_tensor >= args.image_drop_rate + args.text_drop_rate) & (rand_tensor < args.image_drop_rate + args.text_drop_rate + args.cond_drop_rate)
                drop_image_idx = drop_image_idx | drop_both_idx
                drop_text_idx = drop_text_idx | drop_both_idx

                with torch.no_grad():
                    lq_img, gt_img = deg_pipeline(batch["images"], (batch["kernel"], batch["kernel2"], batch["sinc_kernel"]))
                    lq_pt = image_processor(
                        images=lq_img*0.5+0.5,
                        do_rescale=False, return_tensors="pt"
                    ).pixel_values
                    image_embeds = prepare_training_image_embeds(
                        image_encoder, image_processor,
                        ip_adapter_image=lq_pt, ip_adapter_image_embeds=None,
                        device=accelerator.device, drop_rate=args.image_drop_rate, output_hidden_state=args.image_encoder_hidden_feature,
                        idx_to_replace=drop_image_idx
                    )
                    uncond_image_embeds = prepare_training_image_embeds(
                        image_encoder, image_processor,
                        ip_adapter_image=lq_pt, ip_adapter_image_embeds=None,
                        device=accelerator.device, drop_rate=1.0, output_hidden_state=args.image_encoder_hidden_feature,
                        idx_to_replace=torch.ones_like(drop_image_idx)
                    )
                # 1. Load and process the image and text conditioning
                text, orig_size, crop_coords = (
                    batch["text"],
                    batch["original_sizes"],
                    batch["crop_top_lefts"],
                )

                encoded_text = compute_embeddings_fn(text, orig_size, crop_coords)
                uncond_encoded_text = compute_embeddings_fn([""]*len(text), orig_size, crop_coords)

                # encode pixel values with batch size of at most args.vae_encode_batch_size
                gt_img = gt_img.to(dtype=vae.dtype)
                latents = []
                for i in range(0, gt_img.shape[0], args.vae_encode_batch_size):
                    latents.append(vae.encode(gt_img[i : i + args.vae_encode_batch_size]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                # latents = convert_to_latent(gt_img)

                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                bsz = latents.shape[0]
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # 5. Sample a random guidance scale w from U[w_min, w_max]
                # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)

                # 6. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")
                encoded_text["image_embeds"] = image_embeds
                uncond_prompt_embeds = uncond_encoded_text.pop("prompt_embeds")
                uncond_encoded_text["image_embeds"] = image_embeds

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=uncond_prompt_embeds,
                    added_cond_kwargs=uncond_encoded_text,
                ).sample
                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.

                # With the adapters disabled, the `unet` is the regular teacher model.
                accelerator.unwrap_model(unet).disable_adapters()
                with torch.no_grad():

                    # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                    teacher_added_cond = dict()
                    for k,v in encoded_text.items():
                        if isinstance(v, torch.Tensor):
                            teacher_added_cond[k] = v.to(weight_dtype)
                        else:
                            teacher_image_embeds = []
                            for img_emb in v:
                                teacher_image_embeds.append(img_emb.to(weight_dtype))
                            teacher_added_cond[k] = teacher_image_embeds
                    cond_teacher_output = unet(
                        noisy_model_input,
                        start_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=teacher_added_cond,
                    ).sample
                    cond_pred_x0 = get_predicted_original_sample(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    cond_pred_noise = get_predicted_noise(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                    teacher_added_uncond = dict()
                    uncond_encoded_text["image_embeds"] = uncond_image_embeds
                    for k,v in uncond_encoded_text.items():
                        if isinstance(v, torch.Tensor):
                            teacher_added_uncond[k] = v.to(weight_dtype)
                        else:
                            teacher_uncond_image_embeds = []
                            for img_emb in v:
                                teacher_uncond_image_embeds.append(img_emb.to(weight_dtype))
                            teacher_added_uncond[k] = teacher_uncond_image_embeds
                    uncond_teacher_output = unet(
                        noisy_model_input,
                        start_timesteps,
                        encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        added_cond_kwargs=teacher_added_uncond,
                    ).sample
                    uncond_pred_x0 = get_predicted_original_sample(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    uncond_pred_noise = get_predicted_noise(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                    # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                    # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                    # augmented PF-ODE trajectory (solving backward in time)
                    # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index).to(weight_dtype)

                # re-enable unet adapters to turn the `unet` into a student unet.
                accelerator.unwrap_model(unet).enable_adapters()

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # Note that we do not use a separate target network for LCM-LoRA distillation.
                with torch.no_grad():
                    uncond_encoded_text["image_embeds"] = image_embeds
                    target_added_cond = dict()
                    for k,v in uncond_encoded_text.items():
                        if isinstance(v, torch.Tensor):
                            target_added_cond[k] = v.to(weight_dtype)
                        else:
                            target_image_embeds = []
                            for img_emb in v:
                                target_image_embeds.append(img_emb.to(weight_dtype))
                            target_added_cond[k] = target_image_embeds
                    target_noise_pred = unet(
                        x_prev,
                        timesteps,
                        encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        added_cond_kwargs=target_added_cond,
                    ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # 10. Calculate loss
                lcm_loss_arguments = {
                    "target": target.float(),
                    "predict": model_pred.float(),
                }
                loss_dict = dict()
                # total_loss = total_loss + torch.mean(
                #     torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                # )
                # loss_dict["L2Loss"] = total_loss.item()
                for loss_config in lcm_losses:
                    if loss_config.loss.__class__.__name__=="DINOLoss":
                        with torch.no_grad():
                            pixel_target = []
                            latent_target = target.to(dtype=vae.dtype)
                            for i in range(0, latent_target.shape[0], args.vae_encode_batch_size):
                                pixel_target.append(
                                    vae.decode(
                                        latent_target[i : i + args.vae_encode_batch_size] / vae.config.scaling_factor,
                                        return_dict=False
                                    )[0]
                                )
                            pixel_target = torch.cat(pixel_target, dim=0)
                        pixel_pred = []
                        latent_pred = model_pred.to(dtype=vae.dtype)
                        for i in range(0, latent_pred.shape[0], args.vae_encode_batch_size):
                            pixel_pred.append(
                                vae.decode(
                                    latent_pred[i : i + args.vae_encode_batch_size] / vae.config.scaling_factor,
                                    return_dict=False
                                )[0]
                            )
                        pixel_pred = torch.cat(pixel_pred, dim=0)
                        dino_loss_arguments = {
                            "target": pixel_target,
                            "predict": pixel_pred,
                        }
                        non_weighted_loss = loss_config.loss(**dino_loss_arguments, accelerator=accelerator)
                        loss_dict[loss_config.loss.__class__.__name__] = non_weighted_loss.item()
                        total_loss = total_loss + non_weighted_loss * loss_config.weight
                    else:
                        non_weighted_loss = loss_config.loss(**lcm_loss_arguments, accelerator=accelerator)
                        total_loss = total_loss + non_weighted_loss * loss_config.weight
                        loss_dict[loss_config.loss.__class__.__name__] = non_weighted_loss.item()

                # 11. Backpropagate on the online student model (`unet`) (only LoRA)
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        out_images = log_validation(unwrap_model(unet), vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two,
                            lcm_scheduler, image_encoder, image_processor,
                            args, accelerator, weight_dtype, global_step, lq_img, gt_img, is_final_validation=False, log_local=False)

            logs = dict()
            # logs.update({"loss": loss.detach().item()})
            logs.update(loss_dict)
            logs.update({"lr": lr_scheduler.get_last_lr()[0]})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionXLPipeline.save_lora_weights(args.output_dir, unet_lora_layers=unet_lora_state_dict)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        del unet
        torch.cuda.empty_cache()

        # Final inference.
        if args.validation_steps is not None:
            log_validation(unwrap_model(unet), vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                            lcm_scheduler, image_encoder=None, image_processor=None,
                            args=args, accelerator=accelerator, weight_dtype=weight_dtype, step=0, is_final_validation=False, log_local=True)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)