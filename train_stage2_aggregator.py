#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
import argparse
import time
import gc
import logging
import math
import copy
import random
import yaml
import functools
import shutil
import pyrallis
from pathlib import Path
from collections import namedtuple, OrderedDict

import accelerate
import numpy as np
import torch
from safetensors import safe_open
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from data.data_config import DataConfig
from basicsr.utils.degradation_pipeline import RealESRGANDegradation
from losses.loss_config import LossesConfig
from losses.losses import *
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
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from module.aggregator import Aggregator
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
from module.ip_adapter.ip_adapter import MultiIPAdapterImageProjection
from module.ip_adapter.resampler import Resampler
from module.ip_adapter.utils import init_adapter_in_unet, prepare_training_image_embeds
from module.ip_adapter.attention_processor import init_attn_proc
from utils.train_utils import (
    seperate_ip_params_from_unet,
    import_model_class_from_model_name_or_path,
    tensor_to_pil,
    get_train_dataset, prepare_train_dataset, collate_fn,
    encode_prompt, importance_sampling_fn, extract_into_tensor
)
from pipelines.sdxl_instantir import InstantIRPipeline


if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def log_validation(unet, aggregator, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                   scheduler, lcm_scheduler, image_encoder, image_processor, deg_pipeline,
                   args, accelerator, weight_dtype, step, lq_img=None, gt_img=None, is_final_validation=False, log_local=False):
    logger.info("Running validation... ")

    image_logs = []

    # validation_batch = batchify_pil(args.validation_image, args.validation_prompt, deg_pipeline, image_processor)
    lq = [Image.open(lq_example).convert("RGB") for lq_example in args.validation_image]

    pipe = InstantIRPipeline(
            vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            unet, scheduler, aggregator, feature_extractor=image_processor, image_encoder=image_encoder,
    ).to(accelerator.device)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if lq_img is not None and gt_img is not None:
        lq_img = lq_img[:len(args.validation_image)]
        lq_pt = image_processor(
            images=lq_img*0.5+0.5,
            do_rescale=False, return_tensors="pt"
        ).pixel_values
        image = pipe(
            prompt=[""]*len(lq_img),
            image=lq_img,
            ip_adapter_image=lq_pt,
            num_inference_steps=20,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            negative_prompt=[""]*len(lq),
            guidance_scale=5.0,
            height=args.resolution,
            width=args.resolution,
            lcm_scheduler=lcm_scheduler,
        ).images
    else:
        image = pipe(
            prompt=[""]*len(lq),
            image=lq,
            ip_adapter_image=lq,
            num_inference_steps=20,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            negative_prompt=[""]*len(lq),
            guidance_scale=5.0,
            height=args.resolution,
            width=args.resolution,
            lcm_scheduler=lcm_scheduler,
        ).images

    if log_local:
        for i, rec_image in enumerate(image):
            rec_image.save(f"./instantid_{i}.png")
        return

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = [np.asarray(pil_img) for pil_img in image]
            images = np.stack(images, axis=0)
            if lq_img is not None and gt_img is not None:
                input_lq = lq_img.cpu()
                input_lq = np.asarray(input_lq.add(1).div(2).clamp(0, 1))
                input_gt = gt_img.cpu()
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


def remove_attn2(model):
    def recursive_find_module(name, module):
        if not "up_blocks" in name and not "down_blocks" in name and not "mid_block" in name: return
        elif "resnets" in name: return
        if hasattr(module, "attn2"):
            setattr(module, "attn2", None)
            setattr(module, "norm2", None)
            return
        for sub_name, sub_module in module.named_children():
            recursive_find_module(f"{name}.{sub_name}", sub_module)

    for name, module in model.named_children():
        recursive_find_module(name, module)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a IP-Adapter training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to an pretrained controlnet model like tile-controlnet.",
    )
    parser.add_argument(
        "--use_lcm",
        action="store_true",
        help="Whether or not to use lcm unet.",
    )
    parser.add_argument(
        "--pretrained_lcm_lora_path",
        type=str,
        default=None,
        help="Path to LCM lora or model identifier from huggingface.co/models.",
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
        "--aggregator_adapter",
        action="store_true",
        help="Whether or not to add adapter on aggregator.",
    )
    parser.add_argument(
        "--optimize_adapter",
        action="store_true",
        help="Whether or not to optimize IP-Adapter.",
    )
    parser.add_argument(
        "--image_encoder_hidden_feature",
        action="store_true",
        help="Whether or not to use the penultimate hidden states as image embeddings.",
    )
    parser.add_argument(
        "--losses_config_path",
        type=str,
        required=True,
        help=("A yaml file containing losses to use and their weights."),
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stage1_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=3000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--previous_ckpt",
        type=str,
        default=None,
        help=(
            "Whether training should be initialized from a previous checkpoint."
        ),
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--save_only_adapter",
        action="store_true",
        help="Only save extra adapter to save space.",
    )
    parser.add_argument(
        "--cache_prompt_embeds",
        action="store_true",
        help="Whether or not to cache prompt embeds to save memory.",
    )
    parser.add_argument(
        "--importance_sampling",
        action="store_true",
        help="Whether or not to use importance sampling.",
    )
    parser.add_argument(
        "--CFG_scale",
        type=float,
        default=1.0,
        help="CFG for previewer.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
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
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
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
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
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
        "--use_ema_adapter",
        action="store_true",
        help=(
            "use ema ip-adapter for LCM preview"
        ),
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help=(
            "sanity check"
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
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=4000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default='train',
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.sanity_check and args.dataset_name is None and args.train_data_dir is None and args.data_config_path is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.text_drop_rate < 0 or args.text_drop_rate > 1:
        raise ValueError("`--text_drop_rate` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def update_ema_model(ema_model, model, ema_beta):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.copy_(param.detach().lerp(ema_param, ema_beta))


def copy_dict(dict):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
    return new_dict


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

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

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Importance sampling.
    list_of_candidates = np.arange(noise_scheduler.config.num_train_timesteps, dtype='float64')
    prob_dist = importance_sampling_fn(list_of_candidates, noise_scheduler.config.num_train_timesteps, 0.5)
    importance_ratio = prob_dist / prob_dist.sum() * noise_scheduler.config.num_train_timesteps
    importance_ratio = torch.from_numpy(importance_ratio.copy()).float()

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # Text encoder and image encoder.
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_2 = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    # Image processor and image encoder.
    if args.use_clip_encoder:
        image_processor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.feature_extractor_path)
    else:
        image_processor = AutoImageProcessor.from_pretrained(args.feature_extractor_path)
        image_encoder = AutoModel.from_pretrained(args.feature_extractor_path)

    # VAE.
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )

    # UNet.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant
    )

    # Aggregator.
    aggregator = Aggregator.from_unet(unet)
    remove_attn2(aggregator)
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        if args.controlnet_model_name_or_path.endswith(".safetensors"):
            pretrained_cn_state_dict = {}
            with safe_open(args.controlnet_model_name_or_path, framework="pt", device='cpu') as f:
                for key in f.keys():
                    pretrained_cn_state_dict[key] = f.get_tensor(key)
        else:
            pretrained_cn_state_dict = torch.load(os.path.join(args.controlnet_model_name_or_path, "aggregator_ckpt.pt"), map_location="cpu")
        aggregator.load_state_dict(pretrained_cn_state_dict, strict=True)
    else:
        logger.info("Initializing aggregator weights from unet.")

    # Create image embedding projector for IP-Adapters.
    if args.pretrained_adapter_model_path is not None:
        if args.pretrained_adapter_model_path.endswith(".safetensors"):
            pretrained_adapter_state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(args.pretrained_adapter_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        pretrained_adapter_state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        pretrained_adapter_state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            pretrained_adapter_state_dict = torch.load(args.pretrained_adapter_model_path, map_location="cpu")

    # Image embedding Projector.
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

    init_adapter_in_unet(
        unet,
        image_proj_model,
        pretrained_adapter_state_dict,
        adapter_tokens=args.adapter_tokens,
    )

    # EMA adapter for LCM preview.
    if args.use_ema_adapter:
        assert args.optimize_adapter, "No need for EMA with frozen adapter."
        ema_image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.adapter_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
        orig_encoder_hid_proj = unet.encoder_hid_proj
        ema_encoder_hid_proj = MultiIPAdapterImageProjection([ema_image_proj_model])
        orig_attn_procs = unet.attn_processors
        orig_attn_procs_list = torch.nn.ModuleList(orig_attn_procs.values())
        ema_attn_procs = init_attn_proc(unet, args.adapter_tokens, True, True, False)
        ema_attn_procs_list = torch.nn.ModuleList(ema_attn_procs.values())
        ema_attn_procs_list.requires_grad_(False)
        ema_encoder_hid_proj.requires_grad_(False)

        # Initialize EMA state.
        ema_beta = 0.5 ** (args.ema_update_steps / max(args.ema_halflife_steps, 1e-8))
        logger.info(f"Using EMA with beta: {ema_beta}")
        ema_encoder_hid_proj.load_state_dict(orig_encoder_hid_proj.state_dict())
        ema_attn_procs_list.load_state_dict(orig_attn_procs_list.state_dict())

    # Projector for aggregator.
    if args.aggregator_adapter:
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

        init_adapter_in_unet(
            aggregator,
            image_proj_model,
            pretrained_adapter_state_dict,
            adapter_tokens=args.adapter_tokens,
        )
    del pretrained_adapter_state_dict

    # Load LCM LoRA into unet.
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
        logger.info(f"Loaded LCM LoRA with alpha: {args.lora_alpha}")
    # Create LoRA config, FIXME: now hard-coded.
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

    unet.add_adapter(lora_config)
    if args.pretrained_lcm_lora_path is not None:
        incompatible_keys = set_peft_model_state_dict(unet, lora_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if unexpected_keys:
                raise ValueError(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
            for k in missing_keys:
                if "lora" in k:
                    raise ValueError(
                        f"Loading adapter weights from state_dict led to missing keys: {missing_keys}. "
                    )
    lcm_scheduler = LCMSingleStepScheduler.from_config(noise_scheduler.config)

    # Initialize training state.
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    aggregator.requires_grad_(False)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if args.save_only_adapter:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if isinstance(model, Aggregator):
                        torch.save(model.state_dict(), os.path.join(output_dir, "aggregator_ckpt.pt"))
                    weights.pop()

        def load_model_hook(models, input_dir):

            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, Aggregator):
                    aggregator_state_dict = torch.load(os.path.join(input_dir, "aggregator_ckpt.pt"), map_location="cpu")
                    model.load_state_dict(aggregator_state_dict)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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

    if args.gradient_checkpointing:
        aggregator.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(aggregator).dtype != torch.float32:
        raise ValueError(
            f"aggregator loaded as datatype {unwrap_model(aggregator).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

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

    # Optimizer creation
    ip_params, non_ip_params = seperate_ip_params_from_unet(unet)
    if args.optimize_adapter:
        unet_params = ip_params
        unet_frozen_params = non_ip_params
    else: # freeze all unet params
        unet_params = []
        unet_frozen_params = ip_params + non_ip_params
        assert len(unet_frozen_params) == len(list(unet.parameters()))
    params_to_optimize = [p for p in aggregator.parameters()]
    params_to_optimize += unet_params
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Instantiate Loss.
    losses_configs: LossesConfig = pyrallis.load(LossesConfig, open(args.losses_config_path, "r"))
    diffusion_losses = list()
    lcm_losses = list()
    for loss_config in losses_configs.diffusion_losses:
        logger.info(f"Using diffusion loss: {loss_config.name}")
        loss = namedtuple("loss", ["loss", "weight"])
        diffusion_losses.append(
            loss(loss=eval(loss_config.name)(
                visualize_every_k=loss_config.visualize_every_k,
                dtype=weight_dtype,
                accelerator=accelerator,
                **loss_config.init_params), weight=loss_config.weight)
        )
    for loss_config in losses_configs.lcm_losses:
        logger.info(f"Using lcm loss: {loss_config.name}")
        loss = namedtuple("loss", ["loss", "weight"])
        loss_class = eval(loss_config.name)
        lcm_losses.append(loss(loss=loss_class(visualize_every_k=loss_config.visualize_every_k, 
                                                       dtype=weight_dtype,
                                                       accelerator=accelerator,
                                                       dino_model=image_encoder,
                                                       dino_preprocess=image_processor,
                                                       **loss_config.init_params), weight=loss_config.weight))

    # SDXL additional condition that will be added to time embedding.
    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    # Text prompt embeddings.
    @torch.no_grad()
    def compute_embeddings(batch, text_encoders, tokenizers, proportion_empty_prompts=0.0, drop_idx=None, is_train=True):
        prompt_batch = batch[args.caption_column]
        if drop_idx is not None:
            for i in range(len(prompt_batch)):
                prompt_batch[i] = "" if drop_idx[i] else prompt_batch[i]
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, is_train
        )

        add_time_ids = torch.cat(
            [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs

    @torch.no_grad()
    def get_added_cond(batch, prompt_embeds, pooled_prompt_embeds, proportion_empty_prompts=0.0, drop_idx=None, is_train=True):

        add_time_ids = torch.cat(
            [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs

    # Move pixels into latents.
    @torch.no_grad()
    def convert_to_latent(pixels):
        model_input = vae.encode(pixels).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        if args.pretrained_vae_model_name_or_path is None:
            model_input = model_input.to(weight_dtype)
        return model_input

    # Helper functions for training loop.
    # if args.degradation_config_path is not None:
    #     with open(args.degradation_config_path) as stream:
    #         degradation_configs = yaml.safe_load(stream)
    #         deg_pipeline = RealESRGANDegradation(device=accelerator.device, resolution=args.resolution)    
    # else:
    deg_pipeline = RealESRGANDegradation(device=accelerator.device, resolution=args.resolution)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=[text_encoder, text_encoder_2],
        tokenizers=[tokenizer, tokenizer_2],
        is_train=True,
    )

    datasets = []
    datasets_name = []
    datasets_weights = []
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler_train,
        shuffle=True if sampler_train is None else False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    aggregator, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        aggregator, unet, optimizer, train_dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # # cache empty prompts and move text encoders to cpu
    # empty_prompt_embeds, empty_pooled_prompt_embeds = encode_prompt(
    #     [""]*args.train_batch_size, [text_encoder, text_encoder_2], [tokenizer, tokenizer_2], True
    # )
    # compute_embeddings_fn = functools.partial(
    #     get_added_cond,
    #     prompt_embeds=empty_prompt_embeds,
    #     pooled_prompt_embeds=empty_pooled_prompt_embeds,
    #     is_train=True,
    # )
    # text_encoder.to("cpu")
    # text_encoder_2.to("cpu")

    # Move vae, unet and text_encoder to device and cast to `weight_dtype`.
    if args.pretrained_vae_model_name_or_path is None:
        # The VAE is fp32 to avoid NaN losses.
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema_adapter:
        # FIXME: prepare ema model
        # ema_encoder_hid_proj, ema_attn_procs_list = accelerator.prepare(ema_encoder_hid_proj, ema_attn_procs_list)
        ema_encoder_hid_proj.to(accelerator.device)
        ema_attn_procs_list.to(accelerator.device)
    for param in unet_frozen_params:
        param.data = param.data.to(dtype=weight_dtype)
    for param in unet_params:
        param.requires_grad_(True)
    unet.to(accelerator.device)
    aggregator.requires_grad_(True)
    aggregator.to(accelerator.device)
    importance_ratio = importance_ratio.to(accelerator.device)

    # Final sanity check.
    for n, p in unet.named_parameters():
        assert not p.requires_grad, n
        if p.requires_grad:
            assert p.dtype == torch.float32, n
        else:
            assert p.dtype == weight_dtype, n
    for n, p in aggregator.named_parameters():
        if p.requires_grad: assert p.dtype == torch.float32, n
        else:
            raise RuntimeError(f"All parameters in aggregator should require grad. {n}")
            assert p.dtype == weight_dtype, n
    unwrap_model(unet).disable_adapters()

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

        if args.use_ema_adapter:
            unwrap_model(unet).set_attn_processor(ema_attn_procs)
            unwrap_model(unet).encoder_hid_proj = ema_encoder_hid_proj
        batch = next(iter(train_dataloader))
        lq_img, gt_img = deg_pipeline(batch["images"], (batch["kernel"], batch["kernel2"], batch["sinc_kernel"]))
        log_validation(
            unwrap_model(unet), unwrap_model(aggregator), vae,
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            noise_scheduler, lcm_scheduler, image_encoder, image_processor, deg_pipeline,
            args, accelerator, weight_dtype, step=0, lq_img=lq_img, gt_img=gt_img, log_local=True
        )
        if args.use_ema_adapter:
            unwrap_model(unet).set_attn_processor(orig_attn_procs)
            unwrap_model(unet).encoder_hid_proj = orig_encoder_hid_proj
        for n, p in unet.named_parameters():
            if p.requires_grad: assert p.dtype == torch.float32, n
            else: assert p.dtype == weight_dtype, n
        for n, p in aggregator.named_parameters():
            if p.requires_grad: assert p.dtype == torch.float32, n
            else: assert p.dtype == weight_dtype, n
        print("PASS")
        exit()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Optimization steps per epoch = {num_update_steps_per_epoch}")
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

    trainable_models = [aggregator, unet]

    if args.gradient_checkpointing:
        # TODO: add vae
        checkpoint_models = []
    else:
        checkpoint_models = []

    image_logs = None
    tic = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            toc = time.time()
            io_time = toc - tic
            tic = time.time()
            for model in trainable_models + checkpoint_models:
                model.train()
            with accelerator.accumulate(*trainable_models):
                loss = torch.tensor(0.0)

                # Drop conditions.
                rand_tensor = torch.rand(batch["images"].shape[0])
                drop_image_idx = rand_tensor < args.image_drop_rate
                drop_text_idx = (rand_tensor >= args.image_drop_rate) & (rand_tensor < args.image_drop_rate + args.text_drop_rate)
                drop_both_idx = (rand_tensor >= args.image_drop_rate + args.text_drop_rate) & (rand_tensor < args.image_drop_rate + args.text_drop_rate + args.cond_drop_rate)
                drop_image_idx = drop_image_idx | drop_both_idx
                drop_text_idx = drop_text_idx | drop_both_idx

                # Get LQ embeddings
                with torch.no_grad():
                    lq_img, gt_img = deg_pipeline(batch["images"], (batch["kernel"], batch["kernel2"], batch["sinc_kernel"]))
                    lq_pt = image_processor(
                        images=lq_img*0.5+0.5,
                        do_rescale=False, return_tensors="pt"
                    ).pixel_values

                    # Move inputs to latent space.
                    gt_img = gt_img.to(dtype=vae.dtype)
                    lq_img = lq_img.to(dtype=vae.dtype)
                    model_input = convert_to_latent(gt_img)
                    lq_latent = convert_to_latent(lq_img)
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                        lq_latent = lq_latent.to(weight_dtype)

                    # Process conditions.
                    image_embeds = prepare_training_image_embeds(
                        image_encoder, image_processor,
                        ip_adapter_image=lq_pt, ip_adapter_image_embeds=None,
                        device=accelerator.device, drop_rate=args.image_drop_rate, output_hidden_state=args.image_encoder_hidden_feature,
                        idx_to_replace=drop_image_idx
                    )
                    prompt_embeds_input, added_conditions = compute_embeddings_fn(batch, drop_idx=drop_text_idx)

                    # Sample noise that we'll add to the latents.
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    loss_weights = extract_into_tensor(importance_ratio, timesteps, noise.shape) if args.importance_sampling else None

                    if args.CFG_scale > 1.0:
                        # Process negative conditions.
                        drop_idx = torch.ones_like(drop_image_idx)
                        neg_image_embeds = prepare_training_image_embeds(
                            image_encoder, image_processor,
                            ip_adapter_image=lq_pt, ip_adapter_image_embeds=None,
                            device=accelerator.device, drop_rate=1.0, output_hidden_state=args.image_encoder_hidden_feature,
                            idx_to_replace=drop_idx
                        )
                        neg_prompt_embeds_input, neg_added_conditions = compute_embeddings_fn(batch, drop_idx=drop_idx)
                        previewer_model_input = torch.cat([noisy_model_input] * 2)
                        previewer_timesteps = torch.cat([timesteps] * 2)
                        previewer_prompt_embeds = torch.cat([neg_prompt_embeds_input, prompt_embeds_input], dim=0)
                        previewer_added_conditions = {}
                        for k in added_conditions.keys():
                            previewer_added_conditions[k] = torch.cat([neg_added_conditions[k], added_conditions[k]], dim=0)
                        previewer_image_embeds = []
                        for neg_image_embed, image_embed in zip(neg_image_embeds, image_embeds):
                            previewer_image_embeds.append(torch.cat([neg_image_embed, image_embed], dim=0))
                        previewer_added_conditions["image_embeds"] = previewer_image_embeds
                    else:
                        previewer_model_input = noisy_model_input
                        previewer_timesteps = timesteps
                        previewer_prompt_embeds = prompt_embeds_input
                        previewer_added_conditions = {}
                        for k in added_conditions.keys():
                            previewer_added_conditions[k] = added_conditions[k]
                        previewer_added_conditions["image_embeds"] = image_embeds

                    # Get LCM preview latent
                    if args.use_ema_adapter:
                        orig_encoder_hid_proj = unet.encoder_hid_proj
                        orig_attn_procs = unet.attn_processors
                        _ema_attn_procs = copy_dict(ema_attn_procs)
                        unwrap_model(unet).set_attn_processor(_ema_attn_procs)
                        unwrap_model(unet).encoder_hid_proj = ema_encoder_hid_proj
                    unwrap_model(unet).enable_adapters()
                    preview_noise = unet(
                        previewer_model_input, previewer_timesteps,
                        encoder_hidden_states=previewer_prompt_embeds,
                        added_cond_kwargs=previewer_added_conditions,
                        return_dict=False
                    )[0]
                    if args.CFG_scale > 1.0:
                        preview_noise_uncond, preview_noise_cond = preview_noise.chunk(2)
                        cfg_scale = 1.0 + torch.rand_like(timesteps, dtype=weight_dtype) * (args.CFG_scale-1.0)
                        cfg_scale = cfg_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        preview_noise = preview_noise_uncond + cfg_scale * (preview_noise_cond - preview_noise_uncond)
                    preview_latents = lcm_scheduler.step(
                        preview_noise,
                        timesteps,
                        noisy_model_input,
                        return_dict=False
                    )[0]
                    unwrap_model(unet).disable_adapters()
                    if args.use_ema_adapter:
                        unwrap_model(unet).set_attn_processor(orig_attn_procs)
                        unwrap_model(unet).encoder_hid_proj = orig_encoder_hid_proj
                    preview_error_latent = F.mse_loss(preview_latents, model_input).cpu().item()
                    preview_error_noise = F.mse_loss(preview_noise, noise).cpu().item()

                    # # Add fresh noise
                    # if args.noisy_encoder_input:
                    # aggregator_noise = torch.randn_like(preview_latents)
                    # aggregator_input = noise_scheduler.add_noise(preview_latents, aggregator_noise, timesteps)

                down_block_res_samples, mid_block_res_sample = aggregator(
                    lq_latent,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_input,
                    added_cond_kwargs=added_conditions,
                    controlnet_cond=preview_latents,
                    conditioning_scale=1.0,
                    return_dict=False,
                )

                # UNet denoise.
                added_conditions["image_embeds"] = image_embeds
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_input,
                    added_cond_kwargs=added_conditions,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False
                )[0]

                diffusion_loss_arguments = {
                    "target": noise,
                    "predict": model_pred,
                    "prompt_embeddings_input": prompt_embeds_input,
                    "timesteps": timesteps,
                    "weights": loss_weights,
                }

                loss_dict = dict()
                for loss_config in diffusion_losses:
                    non_weighted_loss = loss_config.loss(**diffusion_loss_arguments, accelerator=accelerator)
                    loss = loss + non_weighted_loss * loss_config.weight
                    loss_dict[loss_config.loss.__class__.__name__] = non_weighted_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            toc = time.time()
            forward_time = toc - tic
            tic = toc

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.ema_update_steps == 0:
                    if args.use_ema_adapter:
                        update_ema_model(ema_encoder_hid_proj, orig_encoder_hid_proj, ema_beta)
                        update_ema_model(ema_attn_procs_list, orig_attn_procs_list, ema_beta)

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
                        image_logs = log_validation(
                            unwrap_model(unet), unwrap_model(aggregator), vae,
                            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                            noise_scheduler, lcm_scheduler, image_encoder, image_processor, deg_pipeline,
                            args, accelerator, weight_dtype, global_step, lq_img.detach().clone(), gt_img.detach().clone()
                        )

            logs = {}
            logs.update(loss_dict)
            logs.update(
                {"preview_error_latent": preview_error_latent, "preview_error_noise": preview_error_noise,
                "lr": lr_scheduler.get_last_lr()[0],
                "forward_time": forward_time, "io_time": io_time}
            )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            tic = time.time()

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(save_path, safe_serialization=False)
        # Run a final round of validation.
        # Setting `vae`, `unet`, and `controlnet` to None to load automatically from `args.output_dir`.
        image_logs = None
        if args.validation_image is not None:
            image_logs = log_validation(
                            unwrap_model(unet), unwrap_model(aggregator), vae,
                            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                            noise_scheduler, lcm_scheduler, image_encoder, image_processor, deg_pipeline,
                            args, accelerator, weight_dtype, global_step,
                        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)