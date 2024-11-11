import os
import argparse
import numpy as np
import torch

from PIL import Image
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from diffusers import DDPMScheduler

from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline


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


def resize_img(input_image, max_side=1024, min_side=768, width=None, height=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    # Prepare output size
    if width is not None and height is not None:
        out_w, out_h = width, height
    elif width is not None:
        out_w = width
        out_h = round(h * width / w)
    elif height is not None:
        out_h = height
        out_w = round(w * height / h)
    else:
        out_w, out_h = w, h

    # Resize input to runtime size
    w, h = out_w, out_h
    if min(w, h) < min_side:
        ratio = min_side / min(w, h)
        w, h = round(ratio * w), round(ratio * h)
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        w, h = round(ratio * w), round(ratio * h)
    # Resize to cope with UNet and VAE operations
    w_resize_new = (w // base_pixel_number) * base_pixel_number
    h_resize_new = (h // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image, (out_w, out_h)


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

    # Load pretrained models.
    pipe = InstantIRPipeline.from_pretrained(
        args.sdxl_path,
        torch_dtype=torch.float16,
    )

    # Image prompt projector.
    print("Loading LQ-Adapter...")
    load_adapter_to_pipe(
        pipe,
        args.adapter_model_path if args.adapter_model_path is not None else os.path.join(args.instantir_path, 'adapter.pt'),
        args.vision_encoder_path,
        use_clip_encoder=args.use_clip_encoder,
    )

    # Prepare previewer
    previewer_lora_path = args.previewer_lora_path if args.previewer_lora_path is not None else args.instantir_path
    if previewer_lora_path is not None:
        lora_alpha = pipe.prepare_previewers(previewer_lora_path)
        print(f"use lora alpha {lora_alpha}")
    pipe.to(device=device, dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_pretrained(args.sdxl_path, subfolder="scheduler")
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

    # Load weights.
    print("Loading checkpoint...")
    pretrained_state_dict = torch.load(os.path.join(args.instantir_path, "aggregator.pt"), map_location="cpu")
    pipe.aggregator.load_state_dict(pretrained_state_dict)
    pipe.aggregator.to(device, dtype=torch.float16)

    #################### Restoration ####################

    post_fix = f"_{args.post_fix}" if args.post_fix else ""
    os.makedirs(f"{args.out_path}/{post_fix}", exist_ok=True)

    processed_imgs = os.listdir(os.path.join(args.out_path, post_fix))
    lq_files = []
    lq_batch = []
    if os.path.isfile(args.test_path):
        all_inputs = [args.test_path.split("/")[-1]]
    else:
        all_inputs = os.listdir(args.test_path)
    all_inputs.sort()
    for file in all_inputs:
        if file in processed_imgs:
            print(f"Skip {file}")
            continue
        lq_batch.append(f"{file}")
        if len(lq_batch) == args.batch_size:
            lq_files.append(lq_batch)
            lq_batch = []

    if len(lq_batch) > 0:
        lq_files.append(lq_batch)

    for lq_batch in lq_files:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        lq = []
        out_sizes = []
        for lq_img in lq_batch:
            if os.path.isfile(args.test_path):
                lq_pil = Image.open(args.test_path)
            else:
                lq_pil = Image.open(os.path.join(args.test_path, lq_img))
            lq_pil, out_size = resize_img(lq_pil.convert("RGB"), width=args.width, height=args.height)
            lq.append(lq_pil)
            out_sizes.append(out_size)
        timesteps = None
        if args.denoising_start < 1000:
            timesteps = [
                i * (args.denoising_start//args.num_inference_steps) + pipe.scheduler.config.steps_offset for i in range(0, args.num_inference_steps)
            ]
            timesteps = timesteps[::-1]
            pipe.scheduler.set_timesteps(args.num_inference_steps, device)
            timesteps = pipe.scheduler.timesteps
        if args.prompt is None or len(args.prompt) == 0:
            prompt = "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, \
                ultra HD, extreme meticulous detailing, skin pore detailing, \
                hyper sharpness, perfect without deformations, \
                taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. "
        else:
            prompt = args.prompt
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = prompt*len(lq)
        if args.neg_prompt is None or len(args.neg_prompt) == 0:
            neg_prompt = "blurry, out of focus, unclear, depth of field, over-smooth, \
                sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, \
                dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, \
                watermark, signature, jpeg artifacts, deformed, lowres"
        else:
            neg_prompt = args.neg_prompt
        if not isinstance(neg_prompt, list):
            neg_prompt = [neg_prompt]
        neg_prompt = neg_prompt*len(lq)
        image = pipe(
            prompt=prompt,
            image=lq,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            timesteps=timesteps,
            negative_prompt=neg_prompt,
            guidance_scale=args.cfg,
            previewer_scheduler=lcm_scheduler,
            preview_start=args.preview_start,
            control_guidance_end=args.creative_start,
        ).images

        for i, (rec_image, out_size) in enumerate(zip(image, out_sizes)):
            rec_image.resize([out_size[0], out_size[1]], Image.BILINEAR).save(f"{args.out_path}/{post_fix}/{lq_batch[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstantIR pipeline")
    parser.add_argument(
        "--sdxl_path",
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
        "--creative_start",
        type=float,
        default=1.0,
        help="Proportion of timesteps for creative restoration. 1.0 means no creative restoration while 0.0 means completely free rendering."
    )
    parser.add_argument(
        "--preview_start",
        type=float,
        default=0.0,
        help="Proportion of timesteps to stop previewing at the begining to enhance fidelity to input."
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
        "--width",
        type=int,
        default=None,
        help="Output image width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height."
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Scale of Classifier-Free-Guidance (CFG).",
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
        "--prompt",
        type=str,
        default='',
        nargs="+",
        help=(
            "A set of prompts for creative restoration. Provide either a matching number of test images,"
            " or a single prompt to be used with all inputs."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default='',
        nargs="+",
        help=(
            "A set of negative prompts for creative restoration. Provide either a matching number of test images,"
            " or a single negative prompt to be used with all inputs."
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
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args, device)