import random
import torch
from collections import namedtuple, OrderedDict
from safetensors import safe_open
from .attention_processor import init_attn_proc
from .ip_adapter import MultiIPAdapterImageProjection
from transformers import (
    AutoModel, AutoImageProcessor,
    CLIPVisionModelWithProjection, CLIPImageProcessor)


def init_ip_adapter_in_unet(
        unet,
        image_proj_model,
        pretrained_model_path_or_dict=None,
        adapter_tokens=16,
        use_lcm=False,
        use_adaln=True,
        use_external_kv=False,
    ):
        if pretrained_model_path_or_dict is not None:
            if not isinstance(pretrained_model_path_or_dict, dict):
                if pretrained_model_path_or_dict.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(pretrained_model_path_or_dict, framework="pt", device=unet.device) as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    state_dict = torch.load(pretrained_model_path_or_dict, map_location=unet.device)
            else:
                state_dict = pretrained_model_path_or_dict
            keys = list(state_dict.keys())
            if "image_proj" not in keys and "ip_adapter" not in keys:
                state_dict = revise_state_dict(state_dict)

        # Creat IP cross-attention in unet.
        attn_procs = init_attn_proc(unet, adapter_tokens, use_lcm, use_adaln, use_external_kv)
        unet.set_attn_processor(attn_procs)

        # Load pretrinaed model if needed.
        if pretrained_model_path_or_dict is not None:
            if "ip_adapter" in state_dict.keys():
                adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
                missing, unexpected = adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
                for mk in missing:
                    if "ln" not in mk:
                        raise ValueError(f"Missing keys in adapter_modules: {missing}")
            if "image_proj" in state_dict.keys():
                image_proj_model.load_state_dict(state_dict["image_proj"])

        # Load image projectors into iterable ModuleList.
        image_projection_layers = []
        image_projection_layers.append(image_proj_model)
        unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)

        # Adjust unet config to handle addtional ip hidden states.
        unet.config.encoder_hid_dim_type = "ip_image_proj"


def load_ip_adapter_to_pipe(
        pipe,
        pretrained_model_path_or_dict,
        image_encoder_path=None,
        feature_extractor_path=None,
        use_dino=False,
        ip_adapter_tokens=16,
        use_lcm=True,
        use_adaln=True,
        low_cpu_mem_usage=True,
    ):

        if not isinstance(pretrained_model_path_or_dict, dict):
            if pretrained_model_path_or_dict.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(pretrained_model_path_or_dict, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(pretrained_model_path_or_dict, map_location=pipe.device)
        else:
            state_dict = pretrained_model_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            state_dict = revise_state_dict(state_dict)

        # load CLIP image encoder here if it has not been registered to the pipeline yet
        if image_encoder_path is not None:
            if isinstance(image_encoder_path, str):
                feature_extractor_path = image_encoder_path if feature_extractor_path is None else feature_extractor_path

                image_encoder_path = AutoModel.from_pretrained(
                    image_encoder_path) if use_dino else \
                        CLIPVisionModelWithProjection.from_pretrained(
                            image_encoder_path)
            image_encoder = image_encoder_path.to(pipe.device, dtype=pipe.dtype)

        if feature_extractor_path is not None:
            if isinstance(feature_extractor_path, str):
                feature_extractor_path = AutoImageProcessor.from_pretrained(feature_extractor_path) \
                    if use_dino else CLIPImageProcessor()
            feature_extractor = feature_extractor_path

        # create image encoder if it has not been registered to the pipeline yet
        if hasattr(pipe, "image_encoder") and getattr(pipe, "image_encoder", None) is None:
            pipe.register_modules(image_encoder=image_encoder)

        # create feature extractor if it has not been registered to the pipeline yet
        if hasattr(pipe, "feature_extractor") and getattr(pipe, "feature_extractor", None) is None:
            pipe.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into unet
        unet = getattr(pipe, pipe.unet_name) if not hasattr(pipe, "unet") else pipe.unet
        attn_procs = init_attn_proc(unet, ip_adapter_tokens, use_lcm, use_adaln)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        missing, _ = adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        if len(missing) > 0:
            raise ValueError(f"Missing keys in adapter_modules: {missing}")

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        image_projection_layer = unet._convert_ip_adapter_image_proj_to_diffusers(
            state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
        )
        image_projection_layers.append(image_projection_layer)

        unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        unet.config.encoder_hid_dim_type = "ip_image_proj"

        unet.to(dtype=pipe.dtype, device=pipe.device)


def load_ip_adapter_to_controlnet_pipe(
        pipe,
        pretrained_model_path_or_dict,
        image_encoder_path=None,
        feature_extractor_path=None,
        use_dino=False,
        ip_adapter_tokens=16,
        use_lcm=True,
        use_adaln=True,
        low_cpu_mem_usage=True,
    ):

        if not isinstance(pretrained_model_path_or_dict, dict):
            if pretrained_model_path_or_dict.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(pretrained_model_path_or_dict, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(pretrained_model_path_or_dict, map_location=pipe.device)
        else:
            state_dict = pretrained_model_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            state_dict = revise_state_dict(state_dict)

        # load CLIP image encoder here if it has not been registered to the pipeline yet
        if image_encoder_path is not None:
            if isinstance(image_encoder_path, str):
                feature_extractor_path = image_encoder_path if feature_extractor_path is None else feature_extractor_path

                image_encoder_path = AutoModel.from_pretrained(
                    image_encoder_path) if use_dino else \
                        CLIPVisionModelWithProjection.from_pretrained(
                            image_encoder_path)
            image_encoder = image_encoder_path.to(pipe.device, dtype=pipe.dtype)

        if feature_extractor_path is not None:
            if isinstance(feature_extractor_path, str):
                feature_extractor_path = AutoImageProcessor.from_pretrained(feature_extractor_path) \
                    if use_dino else CLIPImageProcessor()
            feature_extractor = feature_extractor_path

        # create image encoder if it has not been registered to the pipeline yet
        if hasattr(pipe, "image_encoder") and getattr(pipe, "image_encoder", None) is None:
            pipe.register_modules(image_encoder=image_encoder)

        # create feature extractor if it has not been registered to the pipeline yet
        if hasattr(pipe, "feature_extractor") and getattr(pipe, "feature_extractor", None) is None:
            pipe.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into unet
        unet = getattr(pipe, pipe.unet_name) if not hasattr(pipe, "unet") else pipe.unet
        attn_procs = init_attn_proc(unet, ip_adapter_tokens, use_lcm, use_adaln)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        missing, _ = adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        if len(missing) > 0:
            raise ValueError(f"Missing keys in adapter_modules: {missing}")

        controlnet = getattr(pipe, pipe.controlnet_name) if not hasattr(pipe, "controlnet") else pipe.controlnet
        controlnet_attn_procs = init_attn_proc(controlnet, ip_adapter_tokens, use_lcm, use_adaln)
        controlnet.set_attn_processor(controlnet_attn_procs)
        controlnet_adapter_modules = torch.nn.ModuleList(controlnet.attn_processors.values())
        missing, unexpected = controlnet_adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        if len(missing) > 0:
            raise ValueError(f"Missing keys in adapter_modules: {missing}")
        if len(unexpected) > 0:
            for mk in unexpected:
                layer_id = int(mk.split(".")[0])
                if layer_id < len(controlnet.attn_processors.keys()):
                    raise ValueError(f"Failed to load {unexpected} in controlnet adapter_modules")

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        image_projection_layer = unet._convert_ip_adapter_image_proj_to_diffusers(
            state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
        )
        image_projection_layers.append(image_projection_layer)

        unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        unet.config.encoder_hid_dim_type = "ip_image_proj"

        controlnet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        controlnet.config.encoder_hid_dim_type = "ip_image_proj"

        unet.to(dtype=pipe.dtype, device=pipe.device)
        controlnet.to(dtype=pipe.dtype, device=pipe.device)

def revise_state_dict(old_state_dict_or_path, map_location="cpu"):
    new_state_dict = OrderedDict()
    new_state_dict["image_proj"] = OrderedDict()
    new_state_dict["ip_adapter"] = OrderedDict()
    if isinstance(old_state_dict_or_path, str):
        old_state_dict = torch.load(old_state_dict_or_path, map_location=map_location)
    else:
        old_state_dict = old_state_dict_or_path
    for name, weight in old_state_dict.items():
        if name.startswith("image_proj_model."):
            new_state_dict["image_proj"][name[len("image_proj_model."):]] = weight
        elif name.startswith("adapter_modules."):
            new_state_dict["ip_adapter"][name[len("adapter_modules."):]] = weight
    return new_state_dict


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
def encode_image(image_encoder, feature_extractor, image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        return image_enc_hidden_states
    else:
        if isinstance(image_encoder, CLIPVisionModelWithProjection):
            # CLIP image encoder.
            image_embeds = image_encoder(image).image_embeds
        else:
            # DINO image encoder.
            image_embeds = image_encoder(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds


def prepare_training_image_embeds(
    image_encoder, feature_extractor,
    ip_adapter_image, ip_adapter_image_embeds,
    device, drop_rate, output_hidden_state, idx_to_replace=None
):
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        # if len(ip_adapter_image) != len(unet.encoder_hid_proj.image_projection_layers):
        #     raise ValueError(
        #         f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
        #     )

        image_embeds = []
        for single_ip_adapter_image in ip_adapter_image:
            if idx_to_replace is None:
                idx_to_replace = torch.rand(len(single_ip_adapter_image)) < drop_rate
            zero_ip_adapter_image = torch.zeros_like(single_ip_adapter_image)
            single_ip_adapter_image[idx_to_replace] = zero_ip_adapter_image[idx_to_replace]
            single_image_embeds = encode_image(
                image_encoder, feature_extractor, single_ip_adapter_image, device, 1, output_hidden_state
            )
            single_image_embeds = torch.stack([single_image_embeds], dim=1) # FIXME

            image_embeds.append(single_image_embeds)
    else:
        repeat_dims = [1]
        image_embeds = []
        for single_image_embeds in ip_adapter_image_embeds:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                )
                single_negative_image_embeds = single_negative_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                )
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
            else:
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                )
            image_embeds.append(single_image_embeds)

    return image_embeds