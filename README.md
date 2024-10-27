<div align="center">
<h1>InstantIR: Blind Image Restoration with</br>Instant Generative Reference</h1>

[**Jen-Yuan Huang**](https://jy-joy.github.io)<sup>1&nbsp;2</sup>, [**Haofan Wang**](https://haofanwang.github.io/)<sup>2</sup>, [**Qixun Wang**](https://github.com/wangqixun)<sup>2</sup>, [**Xu Bai**](https://huggingface.co/baymin0220)<sup>2</sup>, Hao Ai<sup>2</sup>, Peng Xing<sup>2</sup>, [**Jen-Tse Huang**](https://penguinnnnn.github.io)<sup>3</sup> <br>

<sup>1</sup>Peking University · <sup>2</sup>InstantX Team · <sup>3</sup>The Chinese University of Hong Kong

<!-- <sup>*</sup>corresponding authors -->

<a href='https://arxiv.org/abs/2410.06551'><img src='https://img.shields.io/badge/arXiv-2410.06551-b31b1b.svg'>
<a href='https://jy-joy.github.io/InstantIR/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://huggingface.co/InstantX/InstantIR'><img src='https://img.shields.io/static/v1?label=Model&message=Huggingface&color=orange'></a> 
<!-- [![GitHub](https://img.shields.io/github/stars/InstantID/InstantID?style=social)](https://github.com/InstantID/InstantID) -->

<!-- <a href='https://huggingface.co/spaces/InstantX/InstantID'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
[![ModelScope](https://img.shields.io/badge/ModelScope-Studios-blue)](https://modelscope.cn/studios/instantx/InstantID/summary)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/InstantX/InstantID) -->

</div>

**InstantIR** is a novel single-image restoration model designed to resurrect your damaged images, delivering extrem-quality yet realistic details. You can further boost **InstantIR** performance with additional text prompts, even achieve customized editing!


<!-- >**Abstract**: <br>
> Handling test-time unknown degradation is the major challenge in Blind Image Restoration (BIR), necessitating high model generalization. An effective strategy is to incorporate prior knowledge, either from human input or generative model. In this paper, we introduce Instant-reference Image Restoration (InstantIR), a novel diffusion-based BIR method which dynamically adjusts generation condition during inference. We first extract a compact representation of the input via a pre-trained vision encoder. At each generation step, this representation is used to decode current diffusion latent and instantiate it in the generative prior. The degraded image is then encoded with this reference, providing robust generation condition. We observe the variance of generative references fluctuate with degradation intensity, which we further leverage as an indicator for developing a sampling algorithm adaptive to input quality. Extensive experiments demonstrate InstantIR achieves state-of-the-art performance and offering outstanding visual quality. Through modulating generative references with textual description, InstantIR can restore extreme degradation and additionally feature creative restoration. -->

<img src='assets/teaser_figure.png'>

## 🔥 News
- **10/15/2024** Code released!

## 📝 TODOs:
- Launch oneline demos.

## ✨ Usage
<!-- ### Online Demo
We provide a Gradio Demo on 🤗, click the button below and have fun with InstantIR! -->

### Running locally
#### 1. Clone this repo and setting up environment
```
git clone github
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
|[instantx/instantir](https://huggingface.co/facebook/dinov2-large) | `hf_hub_download(repo_id="InstantX/InstantIR")`

Make sure to import the package first with `from huggingface_hub import hf_hub_download` if you are using Python script.

#### 3. Inference

You can run InstantIR inference using `infer.sh` with the following arguments specified.

| Argument | Value
| :--- | :----------
|--pretrained_model_name_or_path | Path to your SDXL folder.
|--vision_encoder_path | Path to your DINOv2 folder.
|--instantir_path | Path to your InstantIR folder.
|--test_path | Path to your input data.
|--out_path | Path for InstantIR output.

See `infer.py` for additional config options. 

## ⚙️ Training

### Prepare data

InstantIR is trained on [DIV2K](https://www.kaggle.com/datasets/joe1995/div2k-dataset), [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k), [LSDIR](https://data.vision.ee.ethz.ch/yawli/index.html) and [FFHQ](https://www.kaggle.com/datasets/rahulbhalley/ffhq-1024x1024). We adopt dataset weighting to balance the distribution. You can config their weights in ```config_files/IR_dataset.yaml```. Download these training sets and put them under a same directory, which will be used in the following training configurations.

As described in our paper, the training of InstantIR is conducted in two stages:

### Stage1: Degradation Content Perceptor

Train the DCP module on frozen SDXL. We provide example 🤗 [accelerate](https://huggingface.co/docs/accelerate/index) training script in `train_stage1_adapter.py`. You can launch your own training with `accelerate`:

```
accelerate launch --num_processes <num_of_gpus> train_stage1_adapter.py \
    --output_dir <your/output/path> \
    --train_data_dir <your/data/path> \
    --logging_dir <your/logging/path> \
    --pretrained_model_name_or_path <your/sdxl/path> \
    --feature_extractor_path <your/dinov2/path> \
    --save_only_adapter \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --lr_scheduler cosine \
    --lr_num_cycles 1 \
    --resume_from_checkpoint latest
```

After DCP training, distill the Previewer with DCP in `train_previewer_lora.py`:

```
accelerate launch --num_processes <num_of_gpus> train_previewer_lora.py \
    --output_dir <your/output/path> \
    --train_data_dir <your/data/path> \
    --logging_dir <your/logging/path> \
    --pretrained_model_name_or_path <your/sdxl/path> \
    --feature_extractor_path <your/dinov2/path> \
    --pretrained_adapter_model_path <your/dcp/path> \
    --losses_config_path config_files/losses.yaml \
    --data_config_path config_files/IR_dataset.yaml \
    --save_only_adapter \
    --gradient_checkpointing \
    --num_train_timesteps 1000 \
    --num_ddim_timesteps 50 \
    --lora_alpha 1 \
    --mixed_precision fp16 \
    --train_batch_size 32 \
    --vae_encode_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --lr_scheduler cosine \
    --lr_num_cycles 1 \
    --resume_from_checkpoint latest
```


### Stage2: Latents Aggregator

Finally, train the Aggregator with frozen DCP and Previewer in `train_stage2_aggregator.py`:

```
accelerate launch --num_processes <num_of_gpus> train_stage2_aggregator.py \
    --output_dir <your/output/path> \
    --train_data_dir <your/data/path> \
    --logging_dir <your/logging/path> \
    --pretrained_model_name_or_path <your/sdxl/path> \
    --feature_extractor_path <your/dinov2/path> \
    --pretrained_adapter_model_path <your/dcp/path> \
    --pretrained_lcm_lora_path <your/previewer_lora/path> \
    --losses_config_path config_files/losses.yaml \
    --data_config_path config_files/IR_dataset.yaml \
    --image_drop_rate 0.0 \
    --text_drop_rate 0.85 \
    --cond_drop_rate 0.15 \
    --save_only_adapter \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --lr_scheduler cosine \
    --lr_num_cycles 1 \
    --resume_from_checkpoint latest
```

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