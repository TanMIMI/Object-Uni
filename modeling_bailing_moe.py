# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch BailingMoE model."""
import importlib.metadata
import math
import warnings
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from packaging import version
from torch.nn import LayerNorm, Linear, init
from torch.nn.parameter import Parameter
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    ModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import _is_package_available, is_torch_available
from modeling_utils import build_modality_mask, patch_continuous_features_glm as patch_continuous_features
from packaging.version import Version
import torch.distributed as dist
try:
    from atorch.modules.transformer.rmsnorm import AtorchRmsNorm
except Exception as e:
    print(f"failed to import atorch rmsnorm due to {e}")

    class AtorchRmsNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states.to(torch.float32) * torch.rsqrt(variance + self.eps)
            return (self.weight * hidden_states).to(input_dtype)

from configuration_glm import GLMConfig
from modeling_utils import (
    ACT2FN,
    DynamicYaRNScaledRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    LongGLMMemCache,
    NTKScalingRotaryEmbedding,
    RotaryEmbedding,
    RotaryEmbeddingFor3DRope,
    apply_rotary_pos_emb_index,
)

_CHECKPOINT_FOR_DOC = "antgroup/GLM"
_CONFIG_FOR_DOC = "GLMConfig"
_TOKENIZER_FOR_DOC = "GLMTokenizer"
ROUTER_WARMUP_STEP = [0, int(os.getenv("__ROUTER_WARMUP_STEP__", 0)), 0]


GLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "antgroup/GLM",
]

ROPE_TYPE = os.getenv("ROPE_TYPE", "3D")

logger = logging.get_logger(__name__)


try:
    from atorch.modules.transformer.layers import FlashAttnModule
except (ModuleNotFoundError, ImportError):
    logger.info("atorch FlashAttnModule unavailable, fallback to flash_attn-based implementation")

    class FlashAttnModule(nn.Module):
        def __init__(self, causal=False, attention_dropout=0.0, **kwargs):
            super().__init__()
            self.causal = causal
            self.dropout_p = attention_dropout

        def forward(self, query_layer, key_layer, value_layer, glm_mask=None, additive_mask=None, **kwargs):
            from flash_attn import flash_attn_func

            if glm_mask is not None:
                m = glm_mask.reshape(-1)
                assert m.numel() <= 2 * glm_mask.shape[0] and (m[0::2] == 0).all() and (m[1::2] == 1).all(), (
                    "flash_attn fallback only supports glm_mask=[[0],[1]] (zero-length prefix, fully causal); "
                    "please install atorch for other GLM mask patterns."
                )
            if additive_mask is not None:
                raise NotImplementedError("flash_attn fallback does not support additive_mask")

            return flash_attn_func(
                query_layer, key_layer, value_layer,
                dropout_p=self.dropout_p,
                causal=self.causal,
            )


def is_flash_attn_2_available(min_fa2_version='2.0.4'):
    if not is_torch_available():
        return False

    if is_hpu():
        return True

    if not _is_package_available("flash_attn"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(min_fa2_version)
    elif torch.version.hip:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(min_fa2_version)
    else:
        return False


def is_flash_attn_available():
    if not is_torch_available():
        return False

    if not _is_package_available("flash_attn"):
        return False

    return True


def is_ppu():
    device_name = get_device_name()
    return device_name.lower().startswith("ppu")


def get_device_name():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        device_name = "cpu"

    return device_name


def is_hpu():
    try:
        from atorch.utils.import_util import is_torch_npu_available

        return is_torch_npu_available()
    except (ModuleNotFoundError, ImportError):
        return False


def update_moe_config(config, verbose=True):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    kvs = [
        (config.to_dict(), config),
        (config.moe_config, config.moe_config),
        (config.rope_scaling, config.rope_scaling),
    ]
    for dict_like, obj in kvs:
        if not isinstance(dict_like, dict):
            continue
        for k, v in dict_like.items():
            env_v = os.getenv(k)
            if env_v is not None:
                try:
                    if isinstance(v, bool):
                        env_v = bool(int(env_v))
                    else:
                        env_v = type(v)(env_v)
                except Exception:
                    logger.warning(f"failed to convert config value {env_v} to {type(v)}")
                    if v is not None:
                        continue
                ori_v = None
                if isinstance(obj, dict):
                    ori_v = obj.get(k)
                    obj[k] = env_v
                else:
                    ori_v = getattr(config, k, None)
                    setattr(config, k, env_v)
                if verbose and not rank:
                    if ori_v != env_v:
                        logger.info(f"update config {k}: {ori_v} --> {env_v}")
                    else:
                        logger.info(f"skip config {k}: {env_v}")

    global ROUTER_WARMUP_STEP
    ROUTER_WARMUP_STEP = [0, float(os.getenv("__ROUTER_WARMUP_STEP__", 0)), 0]
    if "fine_grained_factor" in config.moe_config:
        factor = config.moe_config["fine_grained_factor"]
        config.num_experts *= factor

        if config.moe_config.get("additional_shared_experts", False) is False:
            config.num_experts -= config.moe_config["num_shared_experts"]
            config.moe_config["top_k"] -= config.moe_config["num_shared_experts"]
    if "expert_model_parallelism" not in config.moe_config:
        config.moe_config["expert_model_parallelism"] = True
    config.moe_config["all2all_overlap"] = False
    config.moe_config["dispatcher_type"] = "AllToAll"
    if verbose and not rank:
        logger.info(f"last moe config:\n{config}")


def is_ppu_available():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device_name.lower().startswith("ppu")


def get_fa_version():
    if is_ppu_available():
        return 'none'
    try:
        from packaging import version
        import flash_attn

        if hasattr(flash_attn, '__version__') and version.parse(flash_attn.__version__) >= version.parse("2.0.0"):
            return 'fa2'
        return 'fa1'
    except Exception as e:
        print(e)
        return 'none'


def detect_repetition(input_ids, min_repeat_tokens=3):
    finished_indexes = []
    input_id_list = [input_ids[i].cpu().detach().numpy().tolist() for i in range(len(input_ids))]
    for i in range(len(input_ids)):
        finish_flag = False
        for num_repeat_tokens in range(input_ids[i].shape[0] // min_repeat_tokens, min_repeat_tokens - 1, -1):
            if num_repeat_tokens == 1:
                min_repeat_times = 4
            elif num_repeat_tokens <= 3:
                min_repeat_times = 3
            else:
                min_repeat_times = 2
            substr = ','.join([str(int(i)) for i in input_id_list[i][input_ids[i].shape[0] - num_repeat_tokens :]])
            last_start = input_ids[i].shape[0] - (num_repeat_tokens * 2 + 1)
            last_end = input_ids[i].shape[0] - num_repeat_tokens

            correct_end = -1
            flag = True
            for t in range(min_repeat_times):
                last_substr = ','.join([str(int(i)) for i in input_id_list[i][last_start:last_end]])
                if last_substr.endswith(substr):
                    correct_end = last_end
                    last_start = last_end - 2 * num_repeat_tokens - 1
                    last_end = last_end - num_repeat_tokens
                elif last_substr.startswith(substr):
                    correct_end = last_start + num_repeat_tokens
                    last_end = last_start
                    last_start = last_start - num_repeat_tokens - 1
                else:
                    flag = False
                    break
            if flag:
                finished_indexes.append(i)
                input_id_list[i] = input_ids[i][:correct_end]
                finish_flag = True
                break
        if not finish_flag:
            input_id_list[i] = input_ids[i]
    return finished_indexes, input_id_list


def check_gpu_sm75_or_greater():
    if not torch.cuda.is_available():
        return False
    capabilities = torch.cuda.get_device_capability()
    is_sm75 = capabilities[0] == 7 and capabilities[1] == 5
    is_greater_than_sm8x = capabilities[0] >= 8
    if is_sm75 or is_greater_than_sm8x:
        return True
    return False


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, partition_sizes=None, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    if partition_sizes is None:
        last_dim = tensor.dim() - 1
        last_dim_size = divide(tensor.size()[last_dim], num_partitions)
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    else:
        assert isinstance(partition_sizes, (List, Tuple))
        tensor_list = torch.split(tensor, partition_sizes, dim=-1)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class BailingMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BailingMoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class ParallelAttention_router(nn.Module):
    def __init__(self, config):
        super(ParallelAttention_router, self).__init__()
        layer_number = 0
        self.layer_number = max(1, layer_number)

        self.flash_attn_drop = 0.01
        self.hidden_size = config.hidden_size
        self.projection_size = config.num_experts

        self.query = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.projection_size, bias=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        enc_position_ids=None,
        encoder_output=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        is_first_step = False
        before_hidden_states = None

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        b = query_layer.size(0)
        s = query_layer.size(1)
        z = query_layer.size(2)

        query_layer = query_layer.float().view(b, s, z, 1)
        key_layer = key_layer.float().view(b, s, z, 1)
        value_layer = value_layer.float().view(b, s, z, 1)

        attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_layer)

        router_output = attn_output.view(b, s, z)

        return router_output


class GLMBaseRouter(nn.Module):

    def _compute_router_probabilities(self):
        raise NotImplementedError

    def _route_tokens(self):
        raise NotImplementedError

    def _cast_classifier(self):
        raise NotImplementedError


class GLMTopNRouter(GLMBaseRouter):

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        if config.moe_config.get('use_attn_router'):
            self.classifier = ParallelAttention_router(config)
        else:
            self.classifier = nn.Linear(
                config.hidden_size, self.num_experts, bias=config.moe_config.get("router_bias", False)
            )
        self.top_k = config.moe_config.get("top_k", 2)
        self.norm_prob = config.moe_config.get("norm_expert_prob", True)
        self.moe_impl = config.moe_config.get("moe_impl", "atorch")

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.classifier(hidden_states)
        if ROUTER_WARMUP_STEP[1] and ROUTER_WARMUP_STEP[2] < 1:
            rand_router_logits = torch.rand_like(router_logits) * router_logits.mean()
            router_alpha = ROUTER_WARMUP_STEP[-1]

            router_logits = router_alpha * router_logits + (1 - router_alpha) * rand_router_logits

        return router_logits

    def _route_tokens(self, router_logits: torch.Tensor) -> Tuple:
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)

        topk_weight, topk_experts_index = torch.topk(router_probs, self.top_k, dim=-1)
        if self.moe_impl == "atorch":
            if self.norm_prob:
                topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
            return topk_weight, topk_experts_index

        topk_selected_mask = torch.nn.functional.one_hot(topk_experts_index, num_classes=self.num_experts)
        topk_selected_mask = ~topk_selected_mask.sum(dim=-2).bool()
        router_probs = router_probs.masked_fill(topk_selected_mask, 0)

        if self.norm_prob:
            router_probs /= router_probs.sum(dim=-1, keepdim=True)
        router_probs = router_probs.to(router_logits.dtype)

        return router_probs, topk_experts_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        router_logits = self._compute_router_probabilities(hidden_states)

        router_probs, topk_experts_index = self._route_tokens(router_logits)

        return router_probs, router_logits, topk_experts_index


class MLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(
        self,
        hidden_size,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        intermediate_size=None,
        use_swiglu=False,
        use_bias=True,
    ):
        super(MLP, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        intermediate_size = int(intermediate_size)

        if use_swiglu:
            self.dense_h_to_4h = Linear(hidden_size, 2 * intermediate_size)
        else:
            self.dense_h_to_4h = Linear(hidden_size, intermediate_size, bias=use_bias)

        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, bias=use_bias)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = ACT2FN["swiglu" if use_swiglu else "gelu"]
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward_w1_fn(self, x):
        x = self.dense_h_to_4h(x)
        x = self.activation_func(x)
        return x

    def forward_w2_fn(self, x):
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class MLPv2(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        intermediate_size=None,
        use_swiglu=True,
        use_bias=False,
    ):

        super().__init__()
        self.hidden_size = hidden_size
        assert intermediate_size is not None
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=use_bias)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=use_bias)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=use_bias)
        self.act_fn = ACT2FN["silu"]

    def forward_w1_fn(self, x):
        output = self.act_fn(self.gate_proj(x))
        output = torch.cat([output, x], dim=-1)
        return output

    def forward_w2_fn(self, x):
        x = torch.split(x, [self.intermediate_size, self.hidden_size], dim=-1)
        x = x[0] * self.up_proj(x[1])

        x = self.down_proj(x)
        return x

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class GLMSparseMLP(torch.nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(
        self,
        config: GLMConfig,
        hidden_size: int,
        output_dropout_prob: float,
        init_method,
        output_layer_init_method=None,
        intermediate_size: int = None,
        use_swiglu: bool = False,
        use_bias: bool = True,
        expert_class: nn.Module = MLP,
        gen_router_type: str = None,
    ) -> None:
        super().__init__()
        router_type = config.moe_config.get("router_type", "top1")
        self.router_type = router_type if gen_router_type is None else gen_router_type

        if self.router_type == "topN":
            self.router = GLMTopNRouter(config)
        elif self.router_type == "MultiRouter":
            self.image_router = GLMTopNRouter(config)
            self.router = GLMTopNRouter(config)
            self.audio_router = GLMTopNRouter(config)  
        else:
            raise NotImplementedError

        if intermediate_size is None:
            intermediate_size = config.hidden_size * 4
        fine_grained_factor = config.moe_config.get("fine_grained_factor", 1)
        expert_intermediate_size = config.moe_config.get("expert_intermediate_size", None) or int(
            intermediate_size / fine_grained_factor
        )
        self.moe_impl = config.moe_config.get("moe_impl", "atorch")
        self.moe_expert_model_parallelism = config.moe_config.get("expert_model_parallelism", False)
        self.all2all_overlap = config.moe_config.get("all2all_overlap", True) and self.moe_expert_model_parallelism
        try:
            from atorch.distributed.distributed import parallel_group
        except (ModuleNotFoundError, ImportError):
            parallel_group = None

        if self.moe_impl == "atorch":
            try:
                try:
                    from atorch.modules.moe._grouped_gemm_moe import Grouped_GEMM_MoE
                except ImportError:
                    from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE
                import atorch
            except (ModuleNotFoundError, ImportError):
                Grouped_GEMM_MoE = None
            if Grouped_GEMM_MoE is None:
                self.experts = None
            elif Version(atorch.__version__) >= Version('1.4.1rc15'):
                logger.info(f'atorch.__version__:{atorch.__version__}')

                self.experts = Grouped_GEMM_MoE(
                    hidden_size=config.hidden_size,
                    expert_intermediate_size=expert_intermediate_size,
                    output_dropout_prob=output_dropout_prob,
                    num_experts=config.num_experts,
                    topk=int(config.moe_config.get("top_k")),
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                    initializer_range=config.initializer_range,
                    use_expert_parallelism=self.moe_expert_model_parallelism,
                    expert_parallel_group=parallel_group("expert")  if self.moe_expert_model_parallelism else None,
                    transpose_w1=config.moe_config.get("transpose_w1", True),
                    merge_w1_v1=config.moe_config.get("merge_w1_v1", False),
                    implementation_type=config.moe_config.get("atorch_backend", "MegaBlocks"),
                    token_dispatcher_type=config.moe_config.get("dispatcher_type", "AllToAll"),
                    is_scale_gradient=config.moe_config.get("is_scale_gradient", True),
                )
            else:
                self.experts = Grouped_GEMM_MoE(
                    hidden_size=config.hidden_size,
                    expert_intermediate_size=expert_intermediate_size,
                    output_dropout_prob=output_dropout_prob,
                    num_experts=config.num_experts,
                    topk=int(config.moe_config.get("top_k")),
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                    initializer_range=config.initializer_range,
                    use_expert_parallelism=self.moe_expert_model_parallelism,
                    expert_parallel_group=parallel_group("expert") if self.moe_expert_model_parallelism else None,
                    transpose_w1=config.moe_config.get("transpose_w1", True),
                    merge_w1_v1=config.moe_config.get("merge_w1_v1", False),
                    implementation_type=config.moe_config.get("atorch_backend", "MegaBlocks"),
                    token_dispatcher_type=config.moe_config.get("dispatcher_type", "AllToAll"),
                )
        else:
            self.experts = nn.ModuleDict()
            for idx in range(config.num_experts):
                self.experts[f"expert_{idx}"] = expert_class(
                    hidden_size,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    intermediate_size=expert_intermediate_size,
                    use_swiglu=use_swiglu,
                    use_bias=use_bias,
                )

        self.num_shared_experts = config.moe_config.get("num_shared_experts", 0)
        if self.num_shared_experts > 0:
            shared_intermediate_size = expert_intermediate_size * self.num_shared_experts
            self.shared_experts = expert_class(
                hidden_size,
                output_dropout_prob,
                init_method,
                output_layer_init_method=output_layer_init_method,
                intermediate_size=shared_intermediate_size,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
            )
        self.register_lsd_hook()
    
    def register_lsd_hook(self):
        """
        1. convert name
        2. split weight
        """

        def hook(state_dict, prefix, *args):
            if self.moe_impl != "atorch":
                return
            needs_convert = any(("experts.w2" in k or "experts.w1" in k) for k in state_dict)
            if not needs_convert:
                return
            from atorch.distributed.distributed import parallel_group
            for key in list(state_dict.keys()):
                if "experts.w2" in key or "experts.w1" in key:
                    new_key = key.replace("experts.w2", "experts.mlp.w2")
                    new_key = new_key.replace("experts.w1", "experts.mlp.w1")
                    logger.info(f"convert state_dict {key} to {new_key}")
                    weight = state_dict.pop(key)
                    expert_group = parallel_group("expert")
                    world_size = dist.get_world_size(expert_group)
                    rank = dist.get_rank(expert_group)
                    weight_size = weight.shape[0] // world_size
                    new_weight = weight[rank * weight_size:
                                        (rank+1) * weight_size]
                    if "24.mlp.experts.mlp.w1" in new_key:
                        logger.info(f"rank={rank} key={new_key} {new_weight.shape} {torch.flatten(new_weight)[:10]}")
                    logger.info(f"convert done {weight.shape} to {new_weight.shape}")

                    state_dict[new_key] = new_weight
                    
            
        self._register_load_state_dict_pre_hook(hook)
    
    
            
    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor, audio_mask: torch.Tensor):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        if self.router_type == "MultiRouter":
            batch_size, seq_len, dim = hidden_states.shape

            if image_mask is not None:
                if len(image_mask.shape) == 3:
                    assert image_mask.shape[-1] == 1
                elif len(image_mask.shape) == 2:
                    assert image_mask.shape == hidden_states.shape[:2]
                    image_mask = image_mask.unsqueeze(-1)
                else:
                    print('image mask shape, sum', image_mask.shape, image_mask.sum())
                    assert False
            if audio_mask is not None:
                if len(audio_mask.shape) == 3:
                    assert audio_mask.shape[-1] == 1
                elif len(audio_mask.shape) == 2:
                    assert audio_mask.shape == hidden_states.shape[:2]
                    audio_mask = audio_mask.unsqueeze(-1)
                else:
                    print('audio mask shape, sum', audio_mask.shape, audio_mask.sum())
                    assert False

            if image_mask is not None and audio_mask is not None:
                audio_mask = audio_mask.to(image_mask.device)
                assert torch.logical_and(image_mask, audio_mask).sum() == 0
            
            image_router_probs, image_router_logits, image_top1_expert_index = self.image_router(hidden_states)
            audio_router_probs, audio_router_logits, audio_top1_expert_index = self.audio_router(hidden_states)
            router_probs, router_logits, top1_expert_index = self.router(hidden_states)

            if image_mask is not None:
                router_probs = image_router_probs * image_mask + router_probs * torch.logical_not(image_mask)
                router_logits = image_router_logits * image_mask + router_logits * torch.logical_not(image_mask)
                top1_expert_index = image_top1_expert_index * image_mask + top1_expert_index * torch.logical_not(image_mask)

            if audio_mask is not None:
                audio_mask = audio_mask.to(router_probs.device)
                router_probs = audio_router_probs * audio_mask + router_probs * torch.logical_not(audio_mask)
                router_logits = audio_router_logits * audio_mask + router_logits * torch.logical_not(audio_mask)
                top1_expert_index = audio_top1_expert_index * audio_mask + top1_expert_index * torch.logical_not(audio_mask)

        else:
            router_probs, router_logits, top1_expert_index = self.router(hidden_states)
        identify = hidden_states
        if self.moe_impl == "atorch":
            shared_experts_fn = tuple()
            if self.num_shared_experts > 0 and self.all2all_overlap:
                shared_experts_fn = (self.shared_experts.forward_w1_fn, self.shared_experts.forward_w2_fn)
            hidden_states = self.experts(hidden_states, router_probs, top1_expert_index, *shared_experts_fn)
            hidden_states = hidden_states.to(identify.dtype)
        else:
            router_mask = router_probs.bool()

            expert_hidden_states = torch.zeros_like(hidden_states)
            for idx, expert in enumerate(self.experts.values()):
                token_indices = router_mask[:, :, idx]
                if not token_indices.any():
                    continue
                combining_weights = router_probs[:, :, idx][token_indices].unsqueeze(-1)
                expert_output = expert(hidden_states[token_indices])

                current_hidden_states = expert_output * combining_weights

                expert_hidden_states[token_indices] += current_hidden_states

            hidden_states = expert_hidden_states

        if self.num_shared_experts > 0 and (self.moe_impl != "atorch" or not self.all2all_overlap):
            hidden_states = hidden_states + self.shared_experts(identify)

        return hidden_states, (router_logits, top1_expert_index)


class VocabEmbedding(torch.nn.Embedding):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super().__init__(config.vocab_size, config.hidden_size)
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        output = F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return output


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class GLMAttention(torch.nn.Module):
    """self-attention layer for GLM.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
        num_key_value_heads=0,
        rotary_type='none',
        bf16=False,
        use_bias=True,
        use_qkv_bias=False,
    ):
        super(GLMAttention, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else self.num_attention_heads
        self.no_mha = self.num_key_value_heads != self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_scale = attention_scale
        self.rotary_type = rotary_type
        self.bf16 = bf16
        if self.no_mha:
            self.query_key_value = Linear(
                hidden_size,
                hidden_size + 2 * self.hidden_size_per_attention_head * self.num_key_value_heads,
                bias=use_qkv_bias,
            )
        else:
            self.query_key_value = Linear(hidden_size, 3 * hidden_size, bias=use_bias)

        self.rotary_type = rotary_type

        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = Linear(hidden_size, hidden_size, bias=use_bias)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb=None,
        past_key_value=None,
        use_cache=False,
        decoding_kwargs: Optional[Dict] = None,
    ):

        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(
            mixed_x_layer,
            3,
            [
                self.num_attention_heads * self.hidden_size_per_attention_head,
                self.num_key_value_heads * self.hidden_size_per_attention_head,
                self.num_key_value_heads * self.hidden_size_per_attention_head,
            ],
        )

        query_layer = mixed_query_layer.view(
            *mixed_query_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
        )
        key_layer = mixed_key_layer.view(
            *mixed_key_layer.size()[:-1], self.num_key_value_heads, self.hidden_size_per_attention_head
        )
        value_layer = mixed_value_layer.view(
            *mixed_value_layer.size()[:-1], self.num_key_value_heads, self.hidden_size_per_attention_head
        )
        if self.rotary_type in ['full-1d']:
            (cos, sin) = rotary_emb
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), position_ids[:, 1, :].contiguous()

            if cos.device != hidden_states.device:
                cos = cos.to(hidden_states.device)
                sin = sin.to(hidden_states.device)

            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)
            query_layer = query_layer.to(value_layer.dtype)
            key_layer = key_layer.to(value_layer.dtype)
        elif self.rotary_type in ['1d', '2d']:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            (cos, sin) = rotary_emb

            if cos.device != hidden_states.device:
                cos = cos.to(hidden_states.device)
                sin = sin.to(hidden_states.device)

            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), position_ids[:, 1, :].contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            if self.rotary_type in ['2d']:
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=1)

        past_key_value = [key_layer, value_layer] if use_cache else None

        if self.no_mha:
            key_layer = repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = repeat_kv(value_layer, self.num_key_value_groups)

        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        if self.attention_scale > 1.0:
            attention_scores = torch.matmul(
                query_layer / math.sqrt(self.attention_scale),
                key_layer.transpose(-1, -2) / math.sqrt(self.hidden_size_per_attention_head * self.attention_scale),
            )
        else:
            attention_scores = torch.matmul(
                query_layer, key_layer.transpose(-1, -2) / math.sqrt(self.hidden_size_per_attention_head)
            )

        attention_mask = attention_mask.type_as(attention_scores)
        attention_scores = torch.mul(attention_scores, attention_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - attention_mask)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output, past_key_value


class GLMFlashAttention(torch.nn.Module):
    """SelfAttention layer optimized by flash-attention.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
        num_key_value_heads=0,
        rotary_type='none',
        bf16=False,
        use_bias=True,
        use_qkv_bias=False,
    ):
        super(GLMFlashAttention, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else self.num_attention_heads
        self.no_mha = self.num_key_value_heads != self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_scale = attention_scale
        if ROPE_TYPE == '3D':
            self.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}
        if self.no_mha:
            self.query_key_value = Linear(
                hidden_size,
                hidden_size + 2 * self.hidden_size_per_attention_head * self.num_key_value_heads,
                bias=use_qkv_bias,
            )
        else:
            self.query_key_value = Linear(hidden_size, 3 * hidden_size, bias=use_bias)

        self.rotary_type = rotary_type
        self.bf16 = bf16

        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = Linear(hidden_size, hidden_size, bias=use_bias)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        rotary_emb=None,
        past_key_value=None,
        use_cache=False,
        decoding_kwargs: Optional[Dict] = None,
    ):
        breakpoint_type = None
        if attention_mask.dim() == 0 or (
            attention_mask.dim() == 1 and attention_mask.shape[0] == hidden_states.shape[0]
        ):
            breakpoint_type = '1d'
        elif attention_mask.dim() == 4 and attention_mask.shape[-2] == 2:
            breakpoint_type = '2d'

        self.flash_attn = FlashAttnModule(
            causal=True if is_flash_attn_2_available() and breakpoint_type is not None else False,
            attention_dropout=self.attention_dropout.p if self.training else 0.0,
        )

        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(
            mixed_x_layer,
            3,
            [
                self.num_attention_heads * self.hidden_size_per_attention_head,
                self.num_key_value_heads * self.hidden_size_per_attention_head,
                self.num_key_value_heads * self.hidden_size_per_attention_head,
            ],
        )

        query_layer = mixed_query_layer.view(
            *mixed_query_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
        )
        key_layer = mixed_key_layer.view(
            *mixed_key_layer.size()[:-1], self.num_key_value_heads, self.hidden_size_per_attention_head
        )
        value_layer = mixed_value_layer.view(
            *mixed_value_layer.size()[:-1], self.num_key_value_heads, self.hidden_size_per_attention_head
        )

        if ROPE_TYPE == '3D':
            (cos, sin) = rotary_emb
            if cos.device != hidden_states.device:
                cos = cos.to(hidden_states.device)
                sin = sin.to(hidden_states.device)
            query_layer = query_layer.transpose(1, 2)
            key_layer = key_layer.transpose(1, 2)
            query_layer, key_layer = apply_multimodal_rotary_pos_emb(
                query_layer, key_layer, cos, sin,
                mrope_section=self.rope_scaling["mrope_section"],
                )
            query_layer = query_layer.transpose(1, 2).to(value_layer.dtype)
            key_layer = key_layer.transpose(1, 2).to(value_layer.dtype)
        elif self.rotary_type in ['full-1d']:
            (cos, sin) = rotary_emb
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), position_ids[:, 1, :].contiguous()

            if cos.device != hidden_states.device:
                cos = cos.to(hidden_states.device)
                sin = sin.to(hidden_states.device)

            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)
            query_layer = query_layer.to(value_layer.dtype)
            key_layer = key_layer.to(value_layer.dtype)
        elif self.rotary_type in ['1d', '2d']:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            (cos, sin) = rotary_emb

            if cos.device != hidden_states.device:
                cos = cos.to(hidden_states.device)
                sin = sin.to(hidden_states.device)

            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), position_ids[:, 1, :].contiguous()

            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            if self.rotary_type in ['2d']:
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=1)
        past_key_value = [key_layer, value_layer.clone()] if use_cache else None

        if self.no_mha:
            key_layer = repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = repeat_kv(value_layer, self.num_key_value_groups)

        mask_kwargs = dict()
        if breakpoint_type == '2d':
            attention_mask = attention_mask.squeeze(1)
        if is_flash_attn_2_available() and breakpoint_type is not None:
            if attention_mask.dtype != torch.int32:
                attention_mask = attention_mask.to(torch.int32)
            mask_kwargs["glm_mask"] = attention_mask
        else:
            if attention_mask.dtype != query_layer.dtype:
                attention_mask = attention_mask.to(query_layer.dtype)
            if is_hpu():
                mask_kwargs["additive_mask"] = attention_mask
            else:
                mask_kwargs["additive_mask"] = (-65504.0) * (1.0 - attention_mask)
        context_layer = self.flash_attn(
            query_layer,
            key_layer,
            value_layer,
            **mask_kwargs,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output, past_key_value


def get_attn_impl(config):
    if hasattr(config, '_attn_implementation'):
        attn_impl = config._attn_implementation
    else:
        attn_impl = auto_detect_attention_mode()

    return attn_impl


GLM_ATTENTION_CLASSES = {
    "eager": GLMAttention,
    "flash_attention_2": GLMFlashAttention,
}


class GLMBlock(torch.nn.Module):
    """A single layer transformer for GLM.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(
        self,
        config: GLMConfig,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
        num_key_value_heads=0,
        intermediate_size=None,
        rotary_type='none',
        use_rmsnorm=False,
        use_atorch_rmsnorm=False,
        use_swiglu=False,
        use_bias=True,
        use_qkv_bias=False,
        bf16=False,
        layer_idx=None,
        focused_attention: bool = False,
        attention_grouping: Optional[Tuple[int, int]] = (4, 128),
        cache_in_memory: bool = False,
        mlp_version: str = "v1",
        flash_attention: str = "flash_attention_1",
    ):
        super(GLMBlock, self).__init__()
        self.config = config
        self.config._attn_implementation = flash_attention
        self.focused_attention = focused_attention
        self.cache_in_memory = cache_in_memory
        self.is_sparse = config.num_experts > 1
        if self.is_sparse:
            first_k_dense_replace = int(config.moe_config.get("first_k_dense_replace", 0))

        self.layer_idx = layer_idx
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        if use_atorch_rmsnorm:
            LayerNormFunc = AtorchRmsNorm
        elif use_rmsnorm:
            LayerNormFunc = BailingMoeRMSNorm
        else:
            LayerNormFunc = LayerNorm
        self.input_layernorm = LayerNormFunc(hidden_size, eps=layernorm_epsilon)

        attn_impl = get_attn_impl(self.config)
        if attn_impl == "eager":
            if True:
                self.attention = GLMAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    rotary_type=rotary_type,
                    num_key_value_heads=num_key_value_heads,
                    use_bias=use_bias,
                    use_qkv_bias=use_qkv_bias,
                )
        else:
                pass

                self.attention = GLMFlashAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    num_key_value_heads=num_key_value_heads,
                    rotary_type=rotary_type,
                    use_bias=use_bias,
                    use_qkv_bias=use_qkv_bias,
                    bf16=bf16,
                )

        self.post_attention_layernorm = LayerNormFunc(hidden_size, eps=layernorm_epsilon)

        mlp_cls = MLPv2 if mlp_version == 'v2' else MLP

        self.use_mot = config.use_mot
        self.gen_router_type = config.gen_router_type

        if not self.is_sparse or layer_idx < first_k_dense_replace:
            self.mlp = mlp_cls(
                hidden_size,
                output_dropout_prob,
                init_method,
                output_layer_init_method=output_layer_init_method,
                intermediate_size=intermediate_size,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
            )
        else:
            use_router = bool(config.use_mot and self.is_sparse and layer_idx >= first_k_dense_replace)

            self.mlp = GLMSparseMLP(
                config,
                hidden_size,
                output_dropout_prob,
                init_method,
                output_layer_init_method=output_layer_init_method,
                intermediate_size=intermediate_size,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
                expert_class=mlp_cls,
                gen_router_type=self.gen_router_type if use_router else None,
            )
            if use_router:
                print("new router initialized!!!")

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        rotary_emb=None,
        image_mask=None,
        audio_mask=None,
        past_key_value=None,
        use_cache=False,
        external_mem_cache=None,
        decoding_kwargs: Optional[Dict] = None,
    ):

        layernorm_output = self.input_layernorm(hidden_states)

        attention_output, present_key_value = self.attention(
            layernorm_output,
            position_ids,
            attention_mask,
            rotary_emb=rotary_emb,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present_external_mem_cache = None
        layernorm_input = hidden_states + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output, image_mask, audio_mask)

        if isinstance(mlp_output, tuple):
            mlp_output, router_tuple = mlp_output
        else:
            router_tuple = None

        output = layernorm_input + mlp_output
        outputs = (output, present_key_value, present_external_mem_cache)
        if router_tuple:
            outputs += router_tuple

        return outputs


class GLMStack(torch.nn.Module):
    """GLM transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(
        self,
        config: GLMConfig,
        num_layers,
        hidden_size,
        num_attention_heads,
        max_sequence_length,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        checkpoint_activations,
        checkpoint_num_layers=1,
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        use_scaled_init_for_output_weights=True,
        block_position_encoding=False,
        attention_scale=1.0,
        num_key_value_heads=0,
        max_memory_length=0,
        bf16=False,
        intermediate_size=None,
        rotary_type='none',
        use_rmsnorm=False,
        use_atorch_rmsnorm=False,
        use_swiglu=False,
        use_bias=False,
        use_qkv_bias=False,
        mlp_version='v1',
        focused_attention: bool = False,
        cache_in_memory: bool = False,
        attention_grouping: Optional[Tuple[int, int]] = (4, 128),
    ):
        super(GLMStack, self).__init__()
        self.config = config

        self._attn_impl = get_attn_impl(self.config)

        self.hidden_size = hidden_size
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.focused_attention = focused_attention
        self.cache_in_memory = cache_in_memory
        self.attention_grouping = attention_grouping
        self.bf16 = bf16

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(0.0, init_method_std, num_layers)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.block_position_encoding = block_position_encoding
        self.rotary_type = rotary_type.lower() if rotary_type else rotary_type
        self.rope_theta = getattr(config, 'rope_theta', 10000)
        logger.info(f"GLMStack rope_theta: {self.rope_theta}")

        if '1d' not in self.rotary_type and '2d' not in self.rotary_type:
            if block_position_encoding:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
            else:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)
        else:
            if ROPE_TYPE == '3D':
                self.rotary_emb = RotaryEmbeddingFor3DRope(
                    self.hidden_size
                    // (num_attention_heads * 2 if self.rotary_type != 'full-1d' else num_attention_heads),
                    base=self.rope_theta,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                )

            elif "ntk" in self.rotary_type:
                self.rotary_emb = NTKScalingRotaryEmbedding(
                    self.hidden_size
                    // (num_attention_heads * 2 if self.rotary_type != 'full-1d' else num_attention_heads),
                    base=self.rope_theta,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                    max_embedding_length=4096,
                    scaling_factor=4,
                )
            elif "linear" in self.rotary_type:
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.hidden_size
                    // (num_attention_heads * 2 if self.rotary_type != 'full-1d' else num_attention_heads),
                    base=self.rope_theta,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                    max_embedding_length=4096,
                    scaling_factor=4,
                )
            elif "yarn" in self.rotary_type:
                self.rotary_emb = DynamicYaRNScaledRotaryEmbedding(
                    self.hidden_size // (num_attention_heads * 2),
                    base=self.rope_theta,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                    max_embedding_length=4096,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.hidden_size
                    // (num_attention_heads * 2 if self.rotary_type != 'full-1d' else num_attention_heads),
                    base=self.rope_theta,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                )

        def get_layer(i):

            return GLMBlock(
                config,
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                attention_scale=attention_scale,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rotary_type=self.rotary_type,
                use_rmsnorm=use_rmsnorm,
                use_atorch_rmsnorm=use_atorch_rmsnorm,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
                use_qkv_bias=use_qkv_bias,
                mlp_version=mlp_version,
                bf16=self.bf16,
                layer_idx=i,
                focused_attention=self.focused_attention,
                cache_in_memory=self.cache_in_memory,
                attention_grouping=self.attention_grouping,
            )

        self.layers = torch.nn.ModuleList([get_layer(i) for i in range(num_layers)])

        if use_atorch_rmsnorm:
            LayerNormFunc = AtorchRmsNorm
        elif use_rmsnorm:
            LayerNormFunc = BailingMoeRMSNorm
        else:
            LayerNormFunc = LayerNorm
        self.final_layernorm = LayerNormFunc(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        external_memory_states=None,
        output_hidden_states=False,
        compute_cosine_similarity: bool = False,
        decoding_kwargs: Optional[Dict] = None,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        global ROUTER_WARMUP_STEP
        if ROUTER_WARMUP_STEP[1]:
            ROUTER_WARMUP_STEP[2] = ROUTER_WARMUP_STEP[0] / ROUTER_WARMUP_STEP[1]
            ROUTER_WARMUP_STEP[0] += 1

        batch_size, query_length = hidden_states.size()[:2]
        past_key_value_length = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_key_value_length = past_key_values[0][0].shape[1]

        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            def build_mask_matrix(seq_length, sep, past_key_value_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length), device=attention_mask.device)
                mask = torch.arange(1, m.shape[-1] + 1, device=m.device).reshape(1, -1, 1)
                ids = torch.arange(1, m.shape[-1] + 1, device=m.device).reshape(1, 1, -1).expand(1, m.shape[-1], -1)
                m = (ids <= mask).type_as(m)

                if is_scalar:
                    m[0, :, : int(sep)] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if past_key_value_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat(
                        (hidden_states.new_ones((batch_size, seq_length, past_key_value_length), device=m.device), m),
                        dim=2,
                    )
                m = m.unsqueeze(1)
                return m

            if self._attn_impl != "flash_attention_2":
                attention_mask = build_mask_matrix(query_length, sep, past_key_value_length=past_key_value_length)
            if (
                is_hpu()
                and self._attn_impl == "flash_attention_2"
                and FlashAttnModule is not None
                and hasattr(FlashAttnModule, "npu_cache")
            ):
                if attention_mask.dim() == 0 or (
                    attention_mask.dim() == 1 and attention_mask.shape[0] == hidden_states.shape[0]
                ):
                    if is_scalar:
                        prefix = [attention_mask.cpu().item()]
                    else:
                        prefix = list(attention_mask.cpu().numpy())
                    additive_mask = build_mask_matrix(query_length, sep, past_key_value_length=past_key_value_length)
                    FlashAttnModule.npu_cache["breakpoint_additive_mask"] = additive_mask
                    FlashAttnModule.npu_cache["breakpoint_prefix"] = prefix
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - past_key_value_length :]

            if (
                is_hpu()
                and self._attn_impl == "flash_attention_2"
                and FlashAttnModule is not None
                and hasattr(FlashAttnModule, "npu_cache")
                and attention_mask.dim() == 4
                and attention_mask.shape[-2] == 2
            ):
                try:
                    from atorch.npu.layers import (
                        create_additive_pack_mask_by_startpoint_endpoint_mask,
                        delete_minus_one_in_pack_glm_mask,
                    )

                    cached_mask = []
                    for i in range(batch_size):
                        cleaned_2d_pack_glm_mask = delete_minus_one_in_pack_glm_mask(attention_mask[i].squeeze(0))
                        additive_mask = create_additive_pack_mask_by_startpoint_endpoint_mask(
                            1, query_length, cleaned_2d_pack_glm_mask.unsqueeze(0)
                        )
                        cached_mask.append(additive_mask)
                    cached_mask = torch.cat(cached_mask, dim=0)
                    FlashAttnModule.npu_cache["breakpoint_additive_mask"] = cached_mask
                except (ModuleNotFoundError, ImportError):
                    pass

        rotary_emb = None
        if '1d' not in self.rotary_type and '2d' not in self.rotary_type:
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.position_embeddings(position_ids)

            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(block_position_ids)
                hidden_states = hidden_states + block_position_embeddings
        else:
            if ROPE_TYPE == '3D':
                rotary_cos, rotary_sin = self.rotary_emb(hidden_states, position_ids)
            else:
                rotary_cos, rotary_sin = self.rotary_emb(hidden_states, seq_len=position_ids.max() + 1)
            rotary_emb = (rotary_cos, rotary_sin)

        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            return _hidden_states.detach() if _hidden_states is not None else None

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = None
        external_mem_layers = None
        all_router_tuple = []
        if use_cache:
            next_decoder_cache = []
            if self.focused_attention:
                external_mem_layers = []

        similarities = list() if compute_cosine_similarity else None
        for i, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            args = [hidden_states, position_ids, attention_mask, rotary_emb]

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    return module(*inputs, None)

                return custom_forward

            past_key_value = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None
            external_mem_i_cache = (
                external_memory_states[i] if external_memory_states and len(past_key_values) > 0 else None
            )

            if self.checkpoint_activations and self.training:
                if self.focused_attention:
                    raise NotImplementedError("The focused attention is not supported for gradient checkpointing.")
                args = [hidden_states, position_ids, attention_mask, rotary_emb, image_mask, audio_mask]
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    *args,
                    past_key_value,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    rotary_emb=rotary_emb,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    external_mem_cache=external_mem_i_cache,
                    decoding_kwargs=decoding_kwargs,
                )
            if compute_cosine_similarity:
                with torch.no_grad():
                    similarities.append(
                        F.cosine_similarity(hidden_states[:, 1:, :], layer_outputs[0][:, 1:, :], dim=-1).mean()
                    )
            hidden_states = layer_outputs[0]
            if len(layer_outputs) >= 4:
                all_router_tuple.append(layer_outputs[3:])

            if use_cache:
                if not self.focused_attention:
                    next_decoder_cache.append(layer_outputs[1])
                    if past_key_values is not None and len(past_key_values) > 0:
                        past_key_values[i] = None
                else:
                    external_mem_layers.append(layer_outputs[2])
                    if layer_outputs[1] is not None:
                        next_decoder_cache.append(
                            [
                                check_detach(layer_outputs[1][0]),
                                check_detach(layer_outputs[1][1]),
                            ]
                        )
                    else:
                        next_decoder_cache = None
                    if past_key_values is not None and len(past_key_values) > 0:
                        past_key_values[i] = None
                    if external_memory_states is not None and len(external_memory_states) > 0:
                        external_memory_states[i] = None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = self.final_layernorm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        outputs = (
            output,
            next_cache,
            external_mem_layers,
            all_hidden_states,
        )
        if all_router_tuple:
            outputs += (all_router_tuple,)
        if compute_cosine_similarity:
            with torch.no_grad():
                mean_similarity = torch.stack(similarities).mean()
                outputs += (mean_similarity,)

        return outputs

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    torch.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i].to(mems[i].device)), dim=1)
                )
        return new_mems


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self._no_split_modules = ["GLMBlock"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMModel):
            module.gradient_checkpointing = value
            module.transformer.checkpoint_activations = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        GLMPreTrainedModel.cache_base_model_prefix = GLMPreTrainedModel.base_model_prefix

        return super(GLMPreTrainedModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def _load_pretrained_model(cls, *model_args, **kwargs):
        if hasattr(GLMPreTrainedModel, "cache_base_model_prefix"):
            GLMPreTrainedModel.base_model_prefix = GLMPreTrainedModel.cache_base_model_prefix
            delattr(GLMPreTrainedModel, "cache_base_model_prefix")
        else:
            GLMPreTrainedModel.base_model_prefix = "glm"

        return super()._load_pretrained_model(*model_args, **kwargs)


GLM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~GLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`GLMTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GLM Model transformer outputting raw hidden-states without any specific head on top.",
    GLM_START_DOCSTRING,
)
class GLMModel(GLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        self.word_embeddings = VocabEmbedding(config)
        self.output_hidden_states = config.output_hidden_states
        config.spatial_merge_size = 2
        config.tokens_per_second = 2 
        self.transformer = GLMStack(
            config,
            config.num_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.max_sequence_length,
            config.embedding_dropout_prob,
            config.attention_dropout_prob,
            config.output_dropout_prob,
            config.checkpoint_activations,
            config.checkpoint_num_layers,
            attention_scale=config.attention_scale,
            num_key_value_heads=config.num_key_value_heads,
            block_position_encoding=config.block_position_encoding,
            max_memory_length=config.max_memory_length,
            intermediate_size=config.intermediate_size,
            rotary_type=config.rotary_type,
            use_rmsnorm=config.use_rmsnorm,
            use_atorch_rmsnorm=config.use_atorch_rmsnorm,
            use_swiglu=config.use_swiglu,
            use_bias=config.use_bias,
            use_qkv_bias=config.use_qkv_bias,
            mlp_version=config.mlp_version,
            bf16=config.bf16,
            focused_attention=config.focused_attention,
            cache_in_memory=config.cache_in_memory,
            attention_grouping=config.attention_grouping,
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )

    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, image_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        self.config.image_token_id = image_token_id if image_token_id is not None else self.config.image_patch_token
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        seq_len = input_ids.shape[1]
        is_image_token = (input_ids == self.config.image_token_id).int()

        image_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


        image_router_mask =  (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .to(inputs_embeds.device)
        ) 

        return inputs_embeds, image_router_mask


    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens):
        inputs_embeds = patch_continuous_features(
           input_embeddings=inputs_embeds, placeholder_loc_lens=placeholder_audio_loc_lens,
           encoded_feats=audio_embeds, encoded_feat_lens=audio_embeds_lengths,
        )
        router_mask_audio = build_modality_mask(placeholder_audio_loc_lens, inputs_embeds.shape[:-1])
        return inputs_embeds, router_mask_audio
     
    def prompt_wrap_navit(self, input_ids, query_embeds_image=None, query_embeds_video=None, query_embeds_audio=None,
        query_embeds_audio_lengths=None, placeholder_audio_loc_lens=None, target_embeds=None):
        inputs_embeds = self.word_embeddings(input_ids)
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None:
            return inputs_embeds
      
        batch_size = input_ids.shape[0]

        image_mask = None
        audio_mask = None

        if query_embeds_image is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_image)
        if query_embeds_video is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_video)
        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids, inputs_embeds, query_embeds_audio, query_embeds_audio_lengths, placeholder_audio_loc_lens,
            )

        return inputs_embeds, image_mask, audio_mask
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_embeds_image: Optional[torch.Tensor] = None,
        query_embeds_video: Optional[torch.Tensor] = None,
        query_embeds_audio: Optional[torch.Tensor] = None,
        query_embeds_audio_lengths: Optional[torch.Tensor] = None,
        placeholder_audio_loc_lens: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        external_mems: Optional[LongGLMMemCache] = None,
        use_cache: bool = False,
        output_hidden_states: Optional[bool] = None,
        compute_cosine_similarity: bool = False,
        decoding_kwargs: Optional[Dict] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw_video: Optional[torch.Tensor] = None,
        second_per_grid_ts=None,
        attention_mask_for_mrope=None,
        **kwargs,
    ) -> ModelOutput:
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        if inputs_embeds is not None:
            words_embeddings = inputs_embeds
            input_shape = inputs_embeds.size()[:2]
        else:
            if (query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None) or input_ids.size(1) == 1:
                words_embeddings = self.word_embeddings(input_ids.clip(0, self.word_embeddings.weight.shape[0] - 1))
                input_shape = input_ids.size()

                image_mask = None
                audio_mask = None
 
            else:
                words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                     input_ids.clip(0, self.word_embeddings.weight.shape[0] - 1), query_embeds_image, query_embeds_video, query_embeds_audio,
                     query_embeds_audio_lengths, placeholder_audio_loc_lens, target_embeds,
                )
                
                input_shape = words_embeddings.size()[:2]
  
        embeddings = words_embeddings

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            block_position_ids = torch.zeros(input_shape[-1], dtype=torch.long, device=device)
            position_ids = torch.stack((position_ids, block_position_ids), dim=0).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, device=device)

        if ROPE_TYPE == '3D':
            try:
                position_ids, rope_deltas = get_rope_index(
                    self.config,
                    input_ids, 
                    image_grid_thw, 
                    image_grid_thw_video, 
                    attention_mask_for_mrope, 
                    second_per_grid_ts,
                )
            except:
                position_ids, rope_deltas = get_rope_index_for_interleave(
                    self.config,
                    input_ids,
                    image_grid_thw,
                    image_grid_thw_video,
                    attention_mask_for_mrope,
                    second_per_grid_ts,
                ) 
        

        transformer_output = self.transformer(
            embeddings,
            position_ids,
            attention_mask,
            past_key_values=past_key_values,
            external_memory_states=external_mems,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            compute_cosine_similarity=compute_cosine_similarity,
            decoding_kwargs=decoding_kwargs,
            image_mask=image_mask,
            audio_mask=audio_mask,
            **kwargs,
        )
        last_hidden_states, past_key_values, external_mems, all_hidden_states = transformer_output[:4]
        output_length = 4 + int(compute_cosine_similarity)
        router_tuple = None if len(transformer_output) == output_length else transformer_output[4]
        cosine_similarity = transformer_output[-1] if compute_cosine_similarity else None
        logits = None

        if output_hidden_states:
            all_hidden_states += (last_hidden_states,)

        if self.output_predict:
            last_hidden_states = last_hidden_states.to(self.word_embeddings.weight.device)
            logits = F.linear(last_hidden_states, self.word_embeddings.weight)

        return ModelOutput(
            last_hidden_states=last_hidden_states,
            logits=logits,
            past_key_values=past_key_values,
            external_mems=external_mems,
            hidden_states=all_hidden_states,
            router_tuple=router_tuple,
            cosine_similarity=cosine_similarity,
        )


@add_start_docstrings("""GLM Model transformer for multiple choice classification""", GLM_START_DOCSTRING)
class GLMForConditionalGeneration(GLMPreTrainedModel):

    def __init__(self, config: GLMConfig):
        super().__init__(config)
        update_moe_config(config)

        self.config = config
        self.config.output_predict = False
        self.router_z_loss_alpha = config.moe_config.get("router_z_loss_alpha", 0.001)
        self.router_balance_loss_alpha = config.moe_config.get("router_balance_loss_alpha", 0.001)
        self.router_group_balance_loss_alpha = config.moe_config.get("router_group_balance_loss_alpha", 0.0)
        self.last_logits_l2_alpha = config.last_logits_l2_alpha

        self.quantized = False
        self.focused_attention = config.focused_attention
        self.glm = GLMModel(config)
        self.use_cache = config.use_cache
        self.max_sequence_length = config.max_sequence_length
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.input_matrix = None
        self.post_init()

    def init_input_matrix(self):
        if self.input_matrix is None:
            self.input_matrix = torch.eye(
                self.lm_head.in_features, device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype
            )

    def get_input_embeddings(self):
        return self.glm.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.glm.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _reorder_cache(self, past, beam_idx):
        if past is None:
            return past
        reordered_decoder_past = []
        for layer_past in past:
            reordered_decoder_past += [
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            ]
        return reordered_decoder_past

    def set_max_memory_length(self, max_memory_length):
        if max_memory_length > 0:
            self.glm.transformer.max_memory_length = max_memory_length

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        external_mems: Optional[LongGLMMemCache] = None,
        inner_loop: bool = False,
        compute_cosine_similarity: bool = False,
        decoding_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> ModelOutput:
        seq_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        external_last_hidden_states = None

        model_output = self.glm(
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            external_mems=external_mems,
            compute_cosine_similarity=compute_cosine_similarity,
            decoding_kwargs=decoding_kwargs,
            **kwargs,
        )
        last_hidden_states = model_output.last_hidden_states
        if external_last_hidden_states is not None:
            external_last_hidden_states = external_last_hidden_states.to(last_hidden_states.device)
            last_hidden_states = torch.cat([external_last_hidden_states, last_hidden_states], dim=1)

        if self.config.norm_head:
            self.init_input_matrix()
            unnormed_head = self.lm_head(self.input_matrix).transpose(1, 0)
            head_norm = unnormed_head.norm(dim=0, keepdim=True, p=2)
            normed_head = unnormed_head / (head_norm + 1e-7)
            lm_logits = F.linear(last_hidden_states, normed_head, None)
        else:
            lm_logits = self.lm_head(last_hidden_states)
        if self.config.norm_softmax:
            lm_logits = lm_logits / (torch.std(lm_logits, dim=-1, keepdim=True) + 1e-6)

        cosine_similarity = model_output.cosine_similarity if compute_cosine_similarity else None

        outputs = ModelOutput(
            logits=lm_logits,
            past_key_values=model_output.past_key_values,
            last_hidden_states=last_hidden_states,
            hidden_states=model_output.hidden_states,
            external_mems=model_output.external_mems,
            router_tuple=model_output.router_tuple,
            router_balance_loss_alpha=self.router_balance_loss_alpha,
            router_group_balance_loss_alpha=self.router_group_balance_loss_alpha,
            router_z_loss_alpha=self.router_z_loss_alpha,
            last_logits_l2_alpha=self.last_logits_l2_alpha,
            cosine_similarity=cosine_similarity,
        )

        outputs["loss"] = None

        return outputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], 1)
                        ),
                    ],
                    dim=-1,
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        if self.focused_attention:
            model_kwargs["external_mems"] = outputs.external_mems

        return model_kwargs

    @torch.no_grad()
    def complete(
        self,
        prompt: Union[str, List[str]],
        tokenizer,
        left_truncate: bool = False,
        max_input_length: Optional[int] = None,
        max_output_length: int = -1,
        top_k: int = 1,
        top_p: float = 1.0,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ):
        max_input_length = max_input_length or self.config.max_sequence_length // 2
        max_output_length = (
            self.config.max_sequence_length - max_input_length
            if max_output_length <= 0 or max_output_length is None
            else max_output_length
        )

        input_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        inputs = tokenizer.build_inputs_for_generation(
            input_ids=input_ids,
            max_input_length=max_input_length,
            left_truncate=left_truncate,
            max_output_length=max_output_length,
            rotary_type=self.config.rotary_type,
            **kwargs,
        )
        inputs = inputs.to(self.device)
        special_tokens = [
            tokenizer.eop_token,
            tokenizer.sop_token,
            tokenizer.eos_token,
        ]

        outputs = self.generate(
            **inputs,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
            pad_token_id=tokenizer.eop_token_id,
            **kwargs,
        )

        outputs = outputs.tolist()
        for i in range(len(outputs)):
            sop_index = outputs[i].index(tokenizer.sop_token_id)
            outputs[i] = outputs[i][sop_index + 1 :]
        outputs = [tokenizer.decode(o) for o in outputs]
        outputs = [self._post_process(o, tokenizer, special_tokens=special_tokens) for o in outputs]

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs


from transformers.models.qwen2.modeling_qwen2 import rotate_half

def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = config.spatial_merge_size
        image_token_id = config.image_patch_token
        video_token_id = config.video_patch_token
        image_start_token_id = config.image_start_token
        video_start_token_id = config.video_start_token

        use_abs_time_pos = second_per_grid_ts is not None

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                if image_grid_thw is not None:
                    vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                if video_grid_thw is not None:
                    vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    if use_abs_time_pos:
                        time_tensor = expanded_range * second_per_grid_t * config.tokens_per_second
                        time_tensor_long = time_tensor.long()
                    else:
                        time_tensor_long = expanded_range.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            return get_llm_rope_index(input_ids, attention_mask)


def get_rope_index_for_interleave(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = config.spatial_merge_size
    image_token_id = config.image_patch_token
    video_token_id = config.video_patch_token
    image_start_token_id = config.image_start_token
    video_start_token_id = config.video_start_token

    use_abs_time_pos = second_per_grid_ts is not None

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        batch_image_index, batch_video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        
        for i, input_ids_seq in enumerate(total_input_ids):
            valid_mask = attention_mask[i] == 1
            input_ids_valid = input_ids_seq[valid_mask]

            vision_blocks = []

            if image_grid_thw is not None:
                image_start_positions = torch.where(input_ids_valid == image_start_token_id)[0].tolist()
                for pos in image_start_positions:
                    if batch_image_index < len(image_grid_thw):
                        t, h, w = image_grid_thw[batch_image_index]
                        vision_blocks.append((
                            pos, 
                            'image', 
                            t.item(), 
                            h.item(),
                            w.item(),
                            0.0
                        ))
                        batch_image_index += 1

            if video_grid_thw is not None:
                video_start_positions = torch.where(input_ids_valid == video_start_token_id)[0].tolist()
                for pos in video_start_positions:
                    if batch_video_index < len(video_grid_thw):
                        t, h, w = video_grid_thw[batch_video_index]
                        second_per_grid_t = second_per_grid_ts[batch_video_index] if second_per_grid_ts is not None else 1.0
                        vision_blocks.append((
                            pos, 
                            'video', 
                            t.item(), 
                            h.item(),
                            w.item(),
                            second_per_grid_t
                        ))
                        batch_video_index += 1

            vision_blocks = sorted(vision_blocks, key=lambda x: x[0])

            llm_pos_ids_list = []
            current_pos = 0
            last_max_id = 0

            for start_idx, v_type, t, h, w, second_per_grid_t in vision_blocks:
                if current_pos < start_idx:
                    text_len = start_idx - current_pos
                    st_idx = last_max_id + 1 if llm_pos_ids_list else 0
                    text_pos = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(text_pos)
                    last_max_id = text_pos.max().item()
                    current_pos = start_idx

                st_idx = last_max_id + 1
                start_token_pos = torch.tensor([[st_idx], [st_idx], [st_idx]])
                llm_pos_ids_list.append(start_token_pos)
                last_max_id = st_idx

                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                vision_block_length = t * llm_grid_h * llm_grid_w

                if vision_block_length > 0:
                    range_tensor = torch.arange(t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    
                    if use_abs_time_pos and v_type == 'video':
                        time_tensor = expanded_range * second_per_grid_t * config.tokens_per_second
                        time_tensor_long = time_tensor.long()
                    else:
                        time_tensor_long = expanded_range.long()
                    
                    t_index = time_tensor_long.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(t, llm_grid_h, -1).flatten()
                    
                    st_idx = last_max_id + 1
                    vision_pos = torch.stack([t_index, h_index, w_index]) + st_idx
                    llm_pos_ids_list.append(vision_pos)
                    last_max_id = vision_pos.max().item()
                
                current_pos = start_idx + 1 + vision_block_length

            if current_pos < len(input_ids_valid):
                text_len = len(input_ids_valid) - current_pos
                st_idx = last_max_id + 1
                text_pos = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                llm_pos_ids_list.append(text_pos)
                last_max_id = text_pos.max().item()
            
            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1)

                seq_len = len(input_ids_valid)
                if llm_positions.shape[1] > seq_len:
                    llm_positions = llm_positions[:, :seq_len]
                elif llm_positions.shape[1] < seq_len:
                    padding = torch.zeros(3, seq_len - llm_positions.shape[1],
                                         dtype=llm_positions.dtype, device=llm_positions.device)
                    llm_positions = torch.cat([llm_positions, padding], dim=1)

                assert llm_positions.shape[1] == seq_len, \
                    f"Position id length ({llm_positions.shape[1]}) does not match sequence length ({seq_len})"

                position_ids[:, i, :seq_len] = llm_positions.to(position_ids.device)

            max_pos_id = llm_positions.max() if llm_pos_ids_list else 0
            mrope_position_deltas.append(max_pos_id + 1 - len(input_ids_valid))
        
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        return get_llm_rope_index(input_ids, attention_mask)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section=[16, 24, 24], unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    (https://qwenlm.github.io/blog/qwen2-vl/)."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
