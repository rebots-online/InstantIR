import argparse
import contextlib
import time
import gc
import logging
import math
import os
import random
import jsonlines
import functools
import shutil
import pyrallis
import itertools
from pathlib import Path
from collections import namedtuple, OrderedDict

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from PIL import Image
from losses.losses import *
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    from transformers import PretrainedConfig
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def get_train_dataset(dataset_name, dataset_dir, args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset(
        dataset_name,
        data_dir=dataset_dir,
        cache_dir=os.path.join(dataset_dir, ".cache"),
        num_proc=4,
        split="train",
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        args.image_column = column_names[0]
        logger.info(f"image column defaulting to {column_names[0]}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            logger.warning(f"dataset {dataset_name} has no column {image_column}")

    if args.caption_column is None:
        args.caption_column = column_names[1]
        logger.info(f"caption column defaulting to {column_names[1]}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            logger.warning(f"dataset {dataset_name} has no column {caption_column}")

    if args.conditioning_image_column is None:
        args.conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {column_names[2]}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            logger.warning(f"dataset {dataset_name} has no column {conditioning_image_column}")

    with accelerator.main_process_first():
        train_dataset = dataset.shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset

def prepare_train_dataset(dataset, accelerator, deg_pipeline, centralize=False):

    # Data augmentations.
    hflip = deg_pipeline.augment_opt['use_hflip'] and random.random() < 0.5
    vflip = deg_pipeline.augment_opt['use_rot'] and random.random() < 0.5
    rot90 = deg_pipeline.augment_opt['use_rot'] and random.random() < 0.5
    augment_transforms = []
    if hflip:
        augment_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
    if vflip:
        augment_transforms.append(transforms.RandomVerticalFlip(p=1.0))
    if rot90:
        # FIXME
        augment_transforms.append(transforms.RandomRotation(degrees=(90,90)))
    torch_transforms=[transforms.ToTensor()]
    if centralize:
        # to [-1, 1]
        torch_transforms.append(transforms.Normalize([0.5], [0.5]))

    training_size = deg_pipeline.degrade_opt['gt_size']
    image_transforms = transforms.Compose(augment_transforms)
    train_transforms = transforms.Compose(torch_transforms)
    train_resize = transforms.Resize(training_size, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(training_size)

    def preprocess_train(examples):
        raw_images = []
        for img_data in examples[args.image_column]:
            raw_images.append(Image.open(img_data).convert("RGB"))

        # Image stack.
        images = []
        original_sizes = []
        crop_top_lefts = []
        # Degradation kernels stack.
        kernel = []
        kernel2 = []
        sinc_kernel = []

        for raw_image in raw_images:
            raw_image = image_transforms(raw_image)
            original_sizes.append((raw_image.height, raw_image.width))

            # Resize smaller edge.
            raw_image = train_resize(raw_image)
            # Crop to training size.
            y1, x1, h, w = train_crop.get_params(raw_image, (training_size, training_size))
            raw_image = crop(raw_image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(raw_image)

            images.append(image)
            k, k2, sk = deg_pipeline.get_kernel()
            kernel.append(k)
            kernel2.append(k2)
            sinc_kernel.append(sk)

        examples["images"] = images
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["kernel"] = kernel
        examples["kernel2"] = kernel2
        examples["sinc_kernel"] = sinc_kernel

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset

def collate_fn(examples):
    images = torch.stack([example["images"] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    kernel = torch.stack([example["kernel"] for example in examples])
    kernel = kernel.to(memory_format=torch.contiguous_format).float()
    kernel2 = torch.stack([example["kernel2"] for example in examples])
    kernel2 = kernel2.to(memory_format=torch.contiguous_format).float()
    sinc_kernel = torch.stack([example["sinc_kernel"] for example in examples])
    sinc_kernel = sinc_kernel.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]

    prompts = []
    for example in examples:
        prompts.append(example[args.caption_column]) if args.caption_column in example else prompts.append("")

    return {
        "images": images,
        "text": prompts,
        "kernel": kernel,
        "kernel2": kernel2,
        "sinc_kernel": sinc_kernel,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

def encode_prompt(prompt_batch, text_encoders, tokenizers, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * np.cos(np.pi * t / max_t))

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def tensor_to_pil(images):
    """
    Convert image tensor or a batch of image tensors to PIL image(s).
    """
    images = (images + 1) / 2
    images_np = images.detach().cpu().numpy()
    if images_np.ndim == 4:
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    elif images_np.ndim == 3:
        images_np = np.transpose(images_np, (1, 2, 0))
        images_np = images_np[None, ...]
    images_np = (images_np * 255).round().astype("uint8")
    if images_np.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images_np]
    else:
        pil_images = [Image.fromarray(image[:, :, :3]) for image in images_np]

    return pil_images

def save_np_to_image(img_np, save_dir):
    img_np = np.transpose(img_np, (0, 2, 3, 1))
    img_np = (img_np * 255).astype(np.uint8)
    img_np = Image.fromarray(img_np[0])
    img_np.save(save_dir)


def seperate_SFT_params_from_unet(unet):
    params = []
    non_params = []
    for name, param in unet.named_parameters():
        if "SFT" in name:
            params.append(param)
        else:
            non_params.append(param)
    return params, non_params


def seperate_lora_params_from_unet(unet):
    keys = []
    frozen_keys = []
    for name, param in unet.named_parameters():
        if "lora" in name:
            keys.append(param)
        else:
            frozen_keys.append(param)
    return keys, frozen_keys


def seperate_ip_params_from_unet(unet):
    ip_params = []
    non_ip_params = []
    for name, param in unet.named_parameters():
        if "encoder_hid_proj." in name or "_ip." in name:
            ip_params.append(param)
        elif "attn" in name and "processor" in name:
            if "ip" in name or "ln" in name:
                ip_params.append(param)
        else:
            non_ip_params.append(param)
    return ip_params, non_ip_params


def seperate_ref_params_from_unet(unet):
    ip_params = []
    non_ip_params = []
    for name, param in unet.named_parameters():
        if "encoder_hid_proj." in name or "_ip." in name:
            ip_params.append(param)
        elif "attn" in name and "processor" in name:
            if "ip" in name or "ln" in name:
                ip_params.append(param)
        elif "extract" in name:
            ip_params.append(param)
        else:
            non_ip_params.append(param)
    return ip_params, non_ip_params


def seperate_ip_modules_from_unet(unet):
    ip_modules = []
    non_ip_modules = []
    for name, module in unet.named_modules():
        if "encoder_hid_proj" in name or "attn2.processor" in name:
            ip_modules.append(module)
        else:
            non_ip_modules.append(module)
    return ip_modules, non_ip_modules


def seperate_SFT_keys_from_unet(unet):
    keys = []
    non_keys = []
    for name, param in unet.named_parameters():
        if "SFT" in name:
            keys.append(name)
        else:
            non_keys.append(name)
    return keys, non_keys


def seperate_ip_keys_from_unet(unet):
    keys = []
    non_keys = []
    for name, param in unet.named_parameters():
        if "encoder_hid_proj." in name or "_ip." in name:
            keys.append(name)
        elif "attn" in name and "processor" in name:
            if "ip" in name or "ln" in name:
                keys.append(name)
        else:
            non_keys.append(name)
    return keys, non_keys