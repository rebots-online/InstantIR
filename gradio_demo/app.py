import os
import torch
import numpy as np
import app as gr
from PIL import Image

from diffusers import DDPMScheduler
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline

from huggingface_hub import hf_hub_download

if not os.path.exists("models/adapter.pt"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/adapter.pt", local_dir=".")
if not os.path.exists("models/aggregator.pt"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/aggregator.pt", local_dir=".")
if not os.path.exists("models/previewer_lora_weights.bin"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/previewer_lora_weights.bin", local_dir=".")

device = "cuda" if torch.cuda.is_available() else "cpu"
sdxl_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
dinov2_repo_id = "facebook/dinov2-large"

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# Load pretrained models.
print("Loading SDXL...")
pipe = InstantIRPipeline.from_pretrained(
    sdxl_repo_id,
    torch_dtype=torch_dtype,
)

# Image prompt projector.
print("Loading LQ-Adapter...")
load_adapter_to_pipe(
    pipe,
    "models/adapter.pt",
    dinov2_repo_id,
)

# Prepare previewer
lora_alpha = pipe.prepare_previewers("models")
print(f"use lora alpha {lora_alpha}")
lora_alpha = pipe.prepare_previewers("latent-consistency/lcm-lora-sdxl", use_lcm=True)
print(f"use lora alpha {lora_alpha}")
pipe.to(device=device, dtype=torch_dtype)
pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_repo_id, subfolder="scheduler")
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

pipe.scheduler = DDPMScheduler.from_pretrained(
    sdxl_repo_id,
    subfolder="scheduler"
)
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
# Load weights.
print("Loading checkpoint...")
aggregator_state_dict = torch.load(
    "models/aggregator.pt",
    map_location="cpu"
)
pipe.aggregator.load_state_dict(aggregator_state_dict, strict=True)
pipe.aggregator.to(device=device, dtype=torch_dtype)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

PROMPT = "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, \
ultra HD, extreme meticulous detailing, skin pore detailing, \
hyper sharpness, perfect without deformations, \
taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. "

NEG_PROMPT = "blurry, out of focus, unclear, depth of field, over-smooth, \
sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, \
dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, \
watermark, signature, jpeg artifacts, deformed, lowres"

def unpack_pipe_out(preview_row, index):
    return preview_row[index][0]

def dynamic_preview_slider(sampling_steps):
    print(sampling_steps)
    return gr.Slider(label="Restoration Previews", value=sampling_steps-1, minimum=0, maximum=sampling_steps-1, step=1)

def dynamic_guidance_slider(sampling_steps):
    return gr.Slider(label="Start Free Rendering", value=sampling_steps, minimum=0, maximum=sampling_steps, step=1)

def show_final_preview(preview_row):
    return preview_row[-1][0]

# @spaces.GPU #[uncomment to use ZeroGPU]
@torch.no_grad()
def instantir_restore(
    lq, prompt="", steps=30, cfg_scale=7.0, guidance_end=1.0,
    creative_restoration=False, seed=3407, height=1024, width=1024, preview_start=0.0):
    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "default" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('default')

    if isinstance(guidance_end, int):
        guidance_end = guidance_end / steps
    if isinstance(preview_start, int):
        preview_start = preview_start / steps
    lq = [resize_img(lq.convert("RGB"), size=(width, height))]
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = [
        i * (1000//steps) + pipe.scheduler.config.steps_offset for i in range(0, steps)
    ]
    timesteps = timesteps[::-1]
    start_timestep = timesteps[0]

    prompt = PROMPT if len(prompt)==0 else prompt
    neg_prompt = NEG_PROMPT

    out = pipe(
        prompt=[prompt]*len(lq),
        image=lq,
        num_inference_steps=steps,
        generator=generator,
        timesteps=timesteps,
        negative_prompt=[neg_prompt]*len(lq),
        guidance_scale=cfg_scale,
        control_guidance_end=guidance_end,
        preview_start=preview_start,
        previewer_scheduler=lcm_scheduler,
        return_dict=False,
        save_preview_row=True,
    )
    for i, preview_img in enumerate(out[1]):
        preview_img.append(f"preview_{i}")
    return out[0][0], out[1]

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # InstantIR: Blind Image Restoration with Instant Generative Reference.

    ### **Official 🤗 Gradio demo of [InstantIR](https://arxiv.org/abs/2410.06551).**
    ### **InstantIR can not only help you restore your broken image, but also capable of imaginative re-creation following your text prompts. See advance usage for more details!**
    ## Basic usage: revitalize your image
    1. Upload an image you want to restore;
    2. Optionally, tune the `Steps` `CFG Scale` parameters. Typically higher steps lead to better results, but less than 50 is recommended for efficiency;
    3. Click `InstantIR magic!`.
    """)
    with gr.Row():
        lq_img = gr.Image(label="Low-quality image", type="pil")
        with gr.Column():
            with gr.Row():
                steps = gr.Number(label="Steps", value=30, step=1)
                cfg_scale = gr.Number(label="CFG Scale", value=7.0, step=0.1)
            with gr.Row():
                height = gr.Number(label="Height", value=1024, step=1)
                weight = gr.Number(label="Weight", value=1024, step=1)
                seed = gr.Number(label="Seed", value=42, step=1)
            # guidance_start = gr.Slider(label="Guidance Start", value=1.0, minimum=0.0, maximum=1.0, step=0.05)
            guidance_end = gr.Slider(label="Start Free Rendering", value=30, minimum=0, maximum=30, step=1)
            preview_start = gr.Slider(label="Preview Start", value=0, minimum=0, maximum=30, step=1)
            prompt = gr.Textbox(label="Restoration prompts (Optional)", placeholder="")
            mode = gr.Checkbox(label="Creative Restoration", value=False)
    with gr.Row():
        with gr.Row():
            restore_btn = gr.Button("InstantIR magic!")
            clear_btn = gr.ClearButton()
        index = gr.Slider(label="Restoration Previews", value=29, minimum=0, maximum=29, step=1)
    with gr.Row():
        output = gr.Image(label="InstantIR restored", type="pil")
        preview = gr.Image(label="Preview", type="pil")
    pipe_out = gr.Gallery(visible=False)
    clear_btn.add([lq_img, output, preview])
    restore_btn.click(
        instantir_restore, inputs=[
            lq_img, prompt, steps, cfg_scale, guidance_end,
            mode, seed, height, weight, preview_start,
        ],
        outputs=[output, pipe_out], api_name="InstantIR"
    )
    steps.change(dynamic_guidance_slider, inputs=steps, outputs=guidance_end)
    output.change(dynamic_preview_slider, inputs=steps, outputs=index)
    index.release(unpack_pipe_out, inputs=[pipe_out, index], outputs=preview)
    output.change(show_final_preview, inputs=pipe_out, outputs=preview)
    gr.Markdown(
    """
    ## Advance usage:
    ### Browse restoration variants:
    1. After InstantIR processing, drag the `Restoration Previews` slider to explore other in-progress versions;
    2. If you like one of them, set the `Start Free Rendering` slider to the same value to get a more refined result.
    ### Creative restoration:
    1. Check the `Creative Restoration` checkbox;
    2. Input your text prompts in the `Restoration prompts` textbox;
    3. Set `Start Free Rendering` slider to a medium value (around half of the `steps`) to provide adequate room for InstantIR creation.
    
    ## Examples
    Here are some examplar usage of InstantIR:
    """)
    # examples = gr.Gallery(label="Examples")

    gr.Markdown(
    """
    ## Citation
    If InstantIR is helpful to your work, please cite our paper via:

    ```
    @article{huang2024instantir,
        title={InstantIR: Blind Image Restoration with Instant Generative Reference},
        author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
        journal={arXiv preprint arXiv:2410.06551},
        year={2024}
    }
    ```
    """)

demo.queue().launch()