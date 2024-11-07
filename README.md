<div align="center">
<h1>InstantIR: Blind Image Restoration with</br>Instant Generative Reference</h1>

[**Jen-Yuan Huang**](https://jy-joy.github.io)<sup>1&nbsp;2</sup>, [**Haofan Wang**](https://haofanwang.github.io/)<sup>2</sup>, [**Qixun Wang**](https://github.com/wangqixun)<sup>2</sup>, [**Xu Bai**](https://huggingface.co/baymin0220)<sup>2</sup>, Hao Ai<sup>2</sup>, Peng Xing<sup>2</sup>, [**Jen-Tse Huang**](https://penguinnnnn.github.io)<sup>3</sup> <br>

<sup>1</sup>Peking University · <sup>2</sup>InstantX Team · <sup>3</sup>The Chinese University of Hong Kong

<!-- <sup>*</sup>corresponding authors -->

<a href='https://arxiv.org/abs/2410.06551'><img src='https://img.shields.io/badge/arXiv-2410.06551-b31b1b.svg'>
<a href='https://jy-joy.github.io/InstantIR/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://huggingface.co/InstantX/InstantIR'><img src='https://img.shields.io/static/v1?label=Model&message=Huggingface&color=orange'></a> 
<a href='https://huggingface.co/spaces/JOY-Huang/InstantIR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
<!-- [![ModelScope](https://img.shields.io/badge/ModelScope-Studios-blue)](https://modelscope.cn/studios/instantx/InstantID/summary)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/InstantX/InstantID) -->

</div>

**InstantIR** is a novel single-image restoration model designed to resurrect your damaged images, delivering extrem-quality yet realistic details. You can further boost **InstantIR** performance with additional text prompts, even achieve customized editing!


<!-- >**Abstract**: <br>
> Handling test-time unknown degradation is the major challenge in Blind Image Restoration (BIR), necessitating high model generalization. An effective strategy is to incorporate prior knowledge, either from human input or generative model. In this paper, we introduce Instant-reference Image Restoration (InstantIR), a novel diffusion-based BIR method which dynamically adjusts generation condition during inference. We first extract a compact representation of the input via a pre-trained vision encoder. At each generation step, this representation is used to decode current diffusion latent and instantiate it in the generative prior. The degraded image is then encoded with this reference, providing robust generation condition. We observe the variance of generative references fluctuate with degradation intensity, which we further leverage as an indicator for developing a sampling algorithm adaptive to input quality. Extensive experiments demonstrate InstantIR achieves state-of-the-art performance and offering outstanding visual quality. Through modulating generative references with textual description, InstantIR can restore extreme degradation and additionally feature creative restoration. -->

<img src='assets/teaser_figure.png'>

## 📢 News
- **11/07/2024** 🔥 Our [Space Demo](https://huggingface.co/spaces/JOY-Huang/InstantIR) is online, thanks [HuggingFace🤗](https://huggingface.co)! Play with **InstantIR** and leave your feedbacks!
- **11/03/2024** 🔥 We provide a Gradio launching script for InstantIR, you can now deploy it on your local machine!
- **11/02/2024** 🔥 InstantIR is now compatitble with `diffusers`🧨, you can utilize features from this fascinating package!
- **10/15/2024** 🔥 Code and model released!

## 📝 TODOs:
- [ ] Tutorial video
- [x] Launch online demo
- [x] Remove dependency on local `diffusers`
- [x] Gradio launching script

## ✨ Usage
<!-- ### Online Demo
We provide a Gradio Demo on 🤗, click the button below and have fun with InstantIR! -->

### Quick start
#### 1. Clone this repo and setting up environment
```sh
git clone https://github.com/JY-Joy/InstantIR.git
cd InstantIR
conda create -n instantir python=3.9 -y
conda activate instantir
pip install -r requirements.txt
```

#### 2. Download pre-trained models

InstantIR is built on SDXL and DINOv2. You can download them either directly from 🤗 huggingface or using Python package.

| 🤗 link | Python command
| :--- | :----------
|[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `hf_hub_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0")`
|[facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | `hf_hub_download(repo_id="facebook/dinov2-large")`
|[InstantX/InstantIR](https://huggingface.co/InstantX/InstantIR) | `hf_hub_download(repo_id="InstantX/InstantIR")`

Note: Make sure to import the package first with `from huggingface_hub import hf_hub_download` if you are using Python script.

#### 3. Inference

You can run InstantIR inference using `infer.sh` with the following arguments specified.

```sh
infer.sh \
    --sdxl_path <path_to_SDXL> \
    --vision_encoder_path <path_to_DINOv2> \
    --instantir_path <path_to_InstantIR> \
    --test_path <path_to_input> \
    --out_path <path_to_output>
```

See `infer.py` for more config options. 

#### 4. Using tips

InstantIR is powerful, but with your help it can do better. InstantIR's flexible pipeline makes it tunable to a large extent. Here are some tips we found particularly useful for various cases you may encounter:
- **Over-smoothing**: reduce `--cfg` to 3.0～5.0. Higher CFG scales can sometimes rigid lines or lack of details.
- **Low fidelity**: set `--preview_start` to 0.1~0.4 to preserve fidelity from inputs. The previewer can yield misleading references when input latent is too noisy. In such cases, we suggest to disable the previewer at early timesteps.
- **Local distortions**: set `--creative_start` to 0.6~0.8. This will let InstantIR render freely in the late diffusion process, where the high-frequency details are generated. Smaller `--creative_start` spares more spaces for creative restoration, but will diminish fidelity.
- **Faster inference**: higher `--preview_start` and lower `--creative_start` can both reduce computational costs and accelerate InstantIR inference.

> [!CAUTION]
> These features are training-free and thus experimental. If you would like to try, we suggest to tune these parameters case-by-case.

### Use InstantIR with diffusers 🧨

InstantIR is fully compatible with `diffusers` and is supported by all those powerful features in this package. You can directly load InstantIR via `diffusers` snippet:

```py
# !pip install diffusers opencv-python transformers accelerate
import torch
from PIL import Image

from diffusers import DDPMScheduler
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline

# suppose you have InstantIR weights under ./models
instantir_path = f'./models'

# load pretrained models
pipe = InstantIRPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)

# load adapter
load_adapter_to_pipe(
    pipe,
    f"{instantir_path}/adapter.pt",
    image_encoder_or_path = 'facebook/dinov2-large',
)

# load previewer lora
pipe.prepare_previewers(instantir_path)
pipe.scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder="scheduler")
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

# load aggregator weights
pretrained_state_dict = torch.load(f"{instantir_path}/aggregator.pt")
pipe.aggregator.load_state_dict(pretrained_state_dict)

# send to GPU and fp16
pipe.to(device='cuda', dtype=torch.float16)
pipe.aggregator.to(device='cuda', dtype=torch.float16)
```

Then, you just need to call the `pipe` and InstantIR will handle your image!

```py
# load a broken image
low_quality_image = Image.open('./assets/sculpture.png').convert("RGB")

# InstantIR restoration
image = pipe(
    image=low_quality_image,
    previewer_scheduler=lcm_scheduler,
).images[0]
```

### Deploy local gradio demo

We provide a python script to launch a local gradio demo of InstantIR, with basic and some advanced features implemented. Start by running the following command in your terminal:

```sh
INSTANTIR_PATH=<path_to_InstantIR> python gradio_demo/app.py
```

Then, visit your local demo via your browser at `http://localhost:7860`.


## ⚙️ Training

### Prepare data

InstantIR is trained on [DIV2K](https://www.kaggle.com/datasets/joe1995/div2k-dataset), [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k), [LSDIR](https://data.vision.ee.ethz.ch/yawli/index.html) and [FFHQ](https://www.kaggle.com/datasets/rahulbhalley/ffhq-1024x1024). We adopt dataset weighting to balance the distribution. You can config their weights in ```config_files/IR_dataset.yaml```. Download these training sets and put them under a same directory, which will be used in the following training configurations.

### Two-stage training
As described in our paper, the training of InstantIR is conducted in two stages. We provide corresponding `.sh` training scripts for each stage. Make sure you have the following arguments adapted to your own use case:

| Argument | Value
| :--- | :----------
| `--pretrained_model_name_or_path` | path to your SDXL folder
| `--feature_extractor_path` | path to your DINOv2 folder
| `--train_data_dir` | your training data directory
| `--output_dir` | path to save model weights
| `--logging_dir` | path to save logs
| `<num_of_gpus>` | number of available GPUs

Other training hyperparameters we used in our experiments are provided in the corresponding `.sh` scripts. You can tune them according to your own needs.

## 💪 Community Resoucres
We sincerely appreciate the community's contribution to InstantIR. Here are some excellent works from the community:

### Duplicate Demo
[fffiloni/InstantIR](https://huggingface.co/spaces/fffiloni/InstantIR)

### ComfyUI
[smthemex/ComfyUI_InstantIR_Wrapper](https://github.com/smthemex/ComfyUI_InstantIR_Wrapper)

## 👏 Acknowledgment
Our work is sponsored by [HuggingFace](https://huggingface.co) and [fal.ai](https://fal.ai).

## 🎓 Citation

If InstantIR is helpful to your work, please cite our paper via:

```
@article{huang2024instantir,
  title={InstantIR: Blind Image Restoration with Instant Generative Reference},
  author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
  journal={arXiv preprint arXiv:2410.06551},
  year={2024}
}
```
