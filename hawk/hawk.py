"""PyTorch Hawk model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import RNNCache
from .external import BlockDiagonalLinear, Conv1D, rnn_param_init
from .fused_cross_entropy import fused_cross_entropy
from .scan_fused import fused_linear_scan

# ------
# Config
# ------


@dataclass
class HawkConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    recurrent_size: int
    num_blocks: int
    conv_width: int


# ----
# Init
# ----


def lecun_init(w: torch.Tensor, d_in: int):
    """Lecun init: w ~ N(0,1/sqrt(d_in))"""
    nn.init.normal_(w, 0, std=1.0 / math.sqrt(d_in))


# -------
# Modules
# -------


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLP(nn.Module):
    def __init__(self, config: HawkConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )

        self.resid_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        lecun_init(self.gate_up_proj.weight, self.hidden_size)
        torch.nn.init.zeros_(self.resid_proj.weight) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)

        gate, up = torch.chunk(x, chunks=2, dim=-1)

        return self.resid_proj(F.gelu(gate) * up)


class Hawk(nn.Module):
    def __init__(self, config: HawkConfig, use_cache: bool = False):
        super().__init__()

        self.config = config

        self.input_xy = nn.Linear(
            config.hidden_size, 2 * config.recurrent_size, bias=False
        )

        self.rg_lru_input_gate = BlockDiagonalLinear(
            width=config.recurrent_size, num_blocks=self.config.num_blocks
        )

        self.rg_lru_a_gate = BlockDiagonalLinear(
            width=config.recurrent_size, num_blocks=self.config.num_blocks
        )

        self.rg_lru_a_param = nn.Parameter(
            torch.empty([config.recurrent_size], dtype=torch.float32)
        )

        self.resid_proj = nn.Linear(
            config.recurrent_size, config.hidden_size, bias=False
        )

        self.use_cache = use_cache

        self.conv1d = Conv1D(
            width=config.recurrent_size,
            temporal_width=self.config.conv_width,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.forget_init(self.rg_lru_a_param)

        lecun_init(self.input_xy.weight, self.config.hidden_size)
        torch.nn.init.zeros_(self.resid_proj.weight) 

    def forget_init(self, w: torch.Tensor) -> torch.Tensor:
        """Initializes the `A` parameter of the RG-LRU."""
        return rnn_param_init(w, min_rad=0.9, max_rad=0.999)

    def epilogue(self, gate, h):
        return self.resid_proj(F.gelu(gate) * h)

    def inference_prologue(self, x):
        # inference-only prologue function
        gate_x = torch.sigmoid(self.rg_lru_input_gate(x))
        gate_a = torch.sigmoid(self.rg_lru_a_gate(x))

        log_a = -8.0 * gate_a * nn.functional.softplus(self.rg_lru_a_param.float())
        a = torch.exp(log_a.float())
        a_square = torch.exp(2 * log_a.float())
        gated_x = x * gate_x

        multiplier = torch.sqrt(1 - a_square)

        normalized_x = gated_x * multiplier.to(x.dtype)

        return a, normalized_x

    def forward(
        self,
        x: torch.Tensor,
        layer_past: RNNCache,
    ) -> Tuple[torch.Tensor, Union[RNNCache, None]]:
        has_layer_past = (
            layer_past is not None
            and layer_past.current_cache_size > 0
            and self.use_cache
        )

        xy = self.input_xy(x)

        x, y = torch.chunk(xy, chunks=2, dim=-1)

        if has_layer_past:
            x = torch.concat([layer_past.conv_state, x], dim=1)
            layer_past.update_conv_cache(x[:, -1:, ...])

        else:
            if self.use_cache:
                layer_past.conv_state = x[:, -3:, ...]

        x = self.conv1d(x=x)

        if has_layer_past:
            x = x[:, -1:, ...]

        x_rg_lru = self.rg_lru_input_gate(x)
        a_rg_lru = self.rg_lru_a_gate(x)

        cache = None
        if not has_layer_past:
            h = fused_linear_scan(x, x_rg_lru, a_rg_lru, self.rg_lru_a_param.float())

            if self.use_cache:
                assert h is not None
                layer_past.update_cache(h[:, -1:, :])
                cache = layer_past

        else:
            a, normalized_x = self.inference_prologue(x)

            h = (a * layer_past.recc_state) + normalized_x

            layer_past.update_cache(h[:, -1:, :])
            cache = layer_past

        output = self.epilogue(y, h)

        return output, cache


class RNNLayer(nn.Module):
    def __init__(self, config: HawkConfig, use_cache: bool = False):
        super().__init__()
        self.use_cache = use_cache

        self.recc = Hawk(config, use_cache)

        self.mlp = MLP(config)

        self.input_norm = RMSNorm(config.hidden_size)
        self.post_recurrent_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: RNNCache | None = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, rnn_cache = self.recc(x=hidden_states, layer_past=layer_past)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_recurrent_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, rnn_cache


class HawkModel(nn.Module):
    def __init__(self, config: HawkConfig, use_cache=False):
        super().__init__()
        self.config = config
        self.use_cache: bool = use_cache
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [(RNNLayer(config, use_cache)) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        lecun_init(self.embed_tokens.weight, self.config.hidden_size)
        torch.nn.init.zeros_(self.lm_head.weight) 

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        layer_past: List[RNNCache | None] | None = None,
    ):
        if layer_past is None:
            layer_past = [None] * len(self.layers)  # type: ignore

        rnn_rnn_cache_list = []

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        assert isinstance(layer_past, List)
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, cache = decoder_layer(
                hidden_states,
                layer_past=layer_past[idx],
            )

            rnn_rnn_cache_list.append(cache)

        hidden_states = self.norm(hidden_states)

        if self.use_cache:
            logits = self.lm_head(hidden_states)
            return logits, rnn_rnn_cache_list

        else:
            if labels is not None:

                shift_labels = labels[..., 1:].contiguous()
                shift_x = hidden_states[..., :-1, :].contiguous()
                loss = fused_cross_entropy(
                    self.lm_head.weight,
                    shift_x.view(-1, shift_x.size(-1)),
                    shift_labels.view(-1),
                )
                return loss

            logits = self.lm_head(hidden_states)
            return logits
