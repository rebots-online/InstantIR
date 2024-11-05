# Copy from diffusers.models.attention.py

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm

from module.min_sdxl import LoRACompatibleLinear, LoRALinearLayer


logger = logging.get_logger(__name__)

def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward

def maybe_grad_checkpoint(resnet, attn, hidden_states, temb, encoder_hidden_states, adapter_hidden_states, do_ckpt=True):

    if do_ckpt:
        hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
        hidden_states, extracted_kv = torch.utils.checkpoint.checkpoint(
            create_custom_forward(attn), hidden_states, encoder_hidden_states, adapter_hidden_states, use_reentrant=False
        )
    else:
        hidden_states = resnet(hidden_states, temb)
        hidden_states, extracted_kv = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            adapter_hidden_states=adapter_hidden_states,
        )
    return hidden_states, extracted_kv


def init_lora_in_attn(attn_module, rank: int = 4, is_kvcopy=False):
    # Set the `lora_layer` attribute of the attention-related matrices.

    attn_module.to_k.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank
        )
    )
    attn_module.to_v.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank
        )
    )

    if not is_kvcopy:
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank
            )
        )

        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=rank,
            )
        )

def drop_kvs(encoder_kvs, drop_chance):
    for layer in encoder_kvs:
        len_tokens = encoder_kvs[layer].self_attention.k.shape[1]
        idx_to_keep = (torch.rand(len_tokens) > drop_chance)

        encoder_kvs[layer].self_attention.k = encoder_kvs[layer].self_attention.k[:, idx_to_keep]
        encoder_kvs[layer].self_attention.v = encoder_kvs[layer].self_attention.v[:, idx_to_keep]

    return encoder_kvs

def clone_kvs(encoder_kvs):
    cloned_kvs = {}
    for layer in encoder_kvs:
        sa_cpy = KVCache(k=encoder_kvs[layer].self_attention.k.clone(), 
                         v=encoder_kvs[layer].self_attention.v.clone())

        ca_cpy = KVCache(k=encoder_kvs[layer].cross_attention.k.clone(),
                         v=encoder_kvs[layer].cross_attention.v.clone())

        cloned_layer_cache = AttentionCache(self_attention=sa_cpy, cross_attention=ca_cpy)
        
        cloned_kvs[layer] = cloned_layer_cache

    return cloned_kvs


class KVCache(object):
    def __init__(self, k, v):
        self.k = k
        self.v = v

class AttentionCache(object):
    def __init__(self, self_attention: KVCache, cross_attention: KVCache):
        self.self_attention = self_attention
        self.cross_attention = cross_attention

class KVCopy(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None,
    ):
        super(KVCopy, self).__init__()

        in_dim = cross_attention_dim or inner_dim

        self.to_k = LoRACompatibleLinear(in_dim, inner_dim, bias=False)
        self.to_v = LoRACompatibleLinear(in_dim, inner_dim, bias=False)

    def forward(self, hidden_states):

        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        return KVCache(k=k, v=v)

    def init_kv_copy(self, source_attn):
        with torch.no_grad():
            self.to_k.weight.copy_(source_attn.to_k.weight)
            self.to_v.weight.copy_(source_attn.to_v.weight)


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x
