import os
import copy
import argparse
import numpy as np
import torch
import json

from PIL import Image
from torchvision import transforms
# from basicsr.utils.degradation_pipeline import RealESRGANDegradation
from tqdm import tqdm
from safetensors import safe_open
from peft import LoraConfig, set_peft_model_state_dict
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from diffusers import (
    DDPMScheduler,
    StableDiffusionXLPipeline
)
from diffusers.utils import convert_unet_state_dict_to_peft

from transformers import (
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    AutoImageProcessor, AutoModel
)

from module.ip_adapter.utils import load_ip_adapter_to_pipe, revise_state_dict, init_ip_adapter_in_unet
from module.ip_adapter.resampler import Resampler
from module.aggregator import Aggregator
from pipelines.sdxl_instantir import InstantIRPipeline, PREVIEWER_LORA_MODULES, LCM_LORA_MODUELS


def name_unet_submodules(unet):
    def recursive_find_module(name, module, end=False):
        if end:
            for sub_name, sub_module in module.named_children():
                sub_module.full_name = f"{name}.{sub_name}"
            return
        if not "up_blocks" in name and not "down_blocks" in name and not "mid_block" in name: return
        elif "resnets" in name: return
        for sub_name, sub_module in module.named_children():
            end = True if sub_name == "transformer_blocks" else False
            recursive_find_module(f"{name}.{sub_name}", sub_module, end)

    for name, module in unet.named_children():
        recursive_find_module(name, module)


def tensor_to_pil(images):
    """
    Convert image tensor or a batch of image tensors to PIL image(s).
    """
    images = images.clamp(0, 1)
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


def calc_mean_std(feat, eps=1e-5):
	"""Calculate mean and std for adaptive_instance_normalization.
	Args:
		feat (Tensor): 4D tensor.
		eps (float): A small value added to the variance to avoid
			divide-by-zero. Default: 1e-5.
	"""
	size = feat.size()
	assert len(size) == 4, 'The input feature should be 4D tensor.'
	b, c = size[:2]
	feat_var = feat.view(b, c, -1).var(dim=2) + eps
	feat_std = feat_var.sqrt().view(b, c, 1, 1)
	feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
	return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def main(args, device):

    # image encoder and feature extractor.
    if args.use_clip_encoder:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.vision_encoder_path,
            subfolder="image_encoder",
        )
        image_processor = CLIPImageProcessor()
    else:
        image_encoder = AutoModel.from_pretrained(args.vision_encoder_path)
        image_processor = AutoImageProcessor.from_pretrained(args.vision_encoder_path)
    image_encoder.to(torch.float16)

    # Base models.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        revision=args.revision,
        variant=args.variant
    )
    unet = pipe.unet

    # Aggregator.
    aggregator = Aggregator.from_unet(unet)

    # Image prompt projector.
    print("Loading LQ-Adapter...")
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
    adapter_path = args.adapter_model_path if args.adapter_model_path is not None else os.path.join(args.instantir_path, 'adapter_ckpt.pt')
    init_ip_adapter_in_unet(
        unet,
        image_proj_model,
        adapter_path,
        adapter_tokens=args.adapter_tokens,
        use_lcm=False,
        use_adaln=True,
        use_external_kv=False,
    )

    pipe = InstantIRPipeline(
            pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2,
            unet, aggregator, pipe.scheduler, feature_extractor=image_processor, image_encoder=image_encoder,
    ).to(device)
    if args.previewer_lora_path is not None:
        lora_state_dict, alpha_dict = StableDiffusionXLPipeline.lora_state_dict(args.previewer_lora_path)
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
            lora_alpha = next(iter(alpha_dict.values()))
        else:
            lora_alpha = 1
        print(f"use lora alpha {lora_alpha}")
    # 9. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    lora_config = LoraConfig(
        r=64,
        target_modules=PREVIEWER_LORA_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
    )

    unet.add_adapter(lora_config)
    if args.previewer_lora_path is not None:
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
            # for k in missing_keys:
            #     if "lora" in k:
            #         raise ValueError(
            #             f"Loading adapter weights from state_dict led to missing keys: {missing_keys}. "
            #         )
    unet.to(dtype=torch.float16)
    unet.disable_adapters()
    # pipe.scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

    # Load weights.
    print("Loading checkpoint...")
    pretrained_state_dict = torch.load(os.path.join(args.instantir_path, "aggregator_ckpt.pt"), map_location="cpu")
    aggregator.load_state_dict(pretrained_state_dict, strict=True)
    aggregator.to(dtype=torch.float16)

    #################### Restoration ####################

    # deg_pipeline = RealESRGANDegradation(device=device, resolution=args.resolution)
    transform = transforms.Compose([
        transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Resize(768, interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.CenterCrop(128),
        # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    post_fix = f"_{args.post_fix}" if args.post_fix else ""
    post_fix = args.instantir_path.split("/")[-2]+f"{post_fix}"
    os.makedirs(f"{args.out_path}/{post_fix}", exist_ok=True)

    processed_imgs = os.listdir(os.path.join(args.out_path, post_fix))
    lq_files = []
    lq_batch = []
    for file in os.listdir(os.path.join(args.test_path, 'input')):
        if file in processed_imgs:
            print(f"Skip {file}")
            continue
        lq_batch.append(f"{file}")
        if len(lq_batch) == args.batch_size:
            lq_files.append(lq_batch)
            lq_batch = []

    if len(lq_batch) > 0:
        lq_files.append(lq_batch)

    # prompts = json.load(open(os.path.join(args.test_path, "prompts.json")))

    for lq_batch in lq_files:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        pil_lqs = [Image.open(os.path.join(args.test_path, 'input', file)) for file in lq_batch]
        lq = [lq_pil.convert("RGB") for lq_pil in pil_lqs]
        # prompt = [prompts[file] for file in lq_batch]

        # timesteps = [
        #     i * (args.denoising_start//args.num_inference_steps) + pipe.scheduler.config.steps_offset for i in range(0, args.num_inference_steps)
        # ]
        # timesteps = timesteps[::-1]
        # pipe.scheduler.set_timesteps(args.num_inference_steps, device)
        # timesteps = pipe.scheduler.timesteps
        # start_timestep = timesteps[0]
        lq_pt = [transforms.ToTensor()(lq_pil) for lq_pil in lq]
        # with torch.no_grad():
        #     gt_pt = torch.stack(lq_pt).to(device, dtype=pipe.vae.dtype)
        #     gt_pt = gt_pt * 2.0 - 1.0
        #     gt_latents = pipe.vae.encode(gt_pt).latent_dist.sample()
        #     gt_latents = gt_latents * pipe.vae.config.scaling_factor

        # InstantStyle magic.
        # scale = {
        #     "down": {"block_2": [0.0, 1.0]},
        #     "up": {"block_0": [0.0, 1.0, 0.0]},
        # }
        # pipe.set_ip_adapter_scale(scale)
        # prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
        # neg_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
        prompt = args.prompt
        print(type(prompt))
        prompt = prompt*len(lq)
        neg_prompt = [""]*len(lq)
        image = pipe(
            prompt=prompt,
            image=lq,
            ip_adapter_image=[lq],
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            # timesteps=timesteps,
            controlnet_conditioning_scale=1.0,
            height=args.resolution,
            width=args.resolution,
            # negative_original_size=(256,256),
            # negative_target_size=(args.resolution,args.resolution),
            negative_prompt=neg_prompt,
            guidance_scale=7.0,
            previewer_scheduler=lcm_scheduler,
            return_dict=False,
            # gt_latent = gt_latents,
            # output_type='pt'
        )[0]
        # norm_image = adaptive_instance_normalization(image, torch.stack(lq_pt).to(device, dtype=image.dtype))
        # image = tensor_to_pil(norm_image)

        if args.save_preview_row:
            for i, lcm_image in enumerate(image[1]):
                lcm_image.save(f"./lcm/{i}.png")
        for i, rec_image in enumerate(image):
            rec_image.save(f"{args.out_path}/{post_fix}/{lq_batch[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstantIR pipeline")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--previewer_lora_path",
        type=str,
        default=None,
        help="Path to LCM lora or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--instantir_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained instantir model.",
    )
    parser.add_argument(
        "--vision_encoder_path",
        type=str,
        default='/share/huangrenyuan/model_zoo/vis_backbone/dinov2_large',
        help="Path to image encoder for IP-Adapters or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_model_path",
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
        "--denoising_start",
        type=int,
        default=1000,
        help="Diffusion start timestep."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Diffusion steps."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Number of tokens to use in IP-adapter cross attention mechanism.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Test batch size."
    )
    parser.add_argument(
        "--post_fix",
        type=str,
        default=None,
        help="Subfolder name for restoration output under the output directory.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default='fp16',
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
        "--save_preview_row",
        action="store_true",
        help="Whether or not to save the intermediate lcm outputs.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts for creative restoration. Provide either a matching number of test images,"
            " or a single prompt to be used with all inputs."
        ),
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        required=True,
        help="Test directory.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./output",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main(args, device)