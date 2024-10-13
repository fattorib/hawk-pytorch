"""PyTorch minGRU model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import RNNCache
from .external import Conv1D
from .scan import linear_scan

# ------
# Config
# ------


@dataclass
class GRUConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    recurrent_size: int


# ------
# Helper
# ------


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
    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

        self.resid_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        lecun_init(self.fc_up_proj.weight, self.hidden_size)
        lecun_init(self.resid_proj.weight, self.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_up_proj(x)
        x = F.gelu(x)
        return self.resid_proj(x)


class GRUBlock(nn.Module):
    def __init__(self, config: GRUConfig, use_cache: bool = False):
        super().__init__()

        self.config = config

        self.input_zh = nn.Linear(
            config.hidden_size, 2 * config.recurrent_size, bias=False
        )

        self.resid_proj = nn.Linear(
            config.recurrent_size, config.hidden_size, bias=False
        )

        self.use_cache = use_cache

        self.temporal_width = 4
        self.conv1d = Conv1D(
            width=config.recurrent_size,
            temporal_width=self.temporal_width,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        lecun_init(self.input_zh.weight, self.config.hidden_size)
        lecun_init(self.resid_proj.weight, self.config.recurrent_size)

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

        if has_layer_past:
            x = torch.concat([layer_past.conv_state, x], dim=1)
            layer_past.update_conv_cache(x[:, -1:, ...])

        else:
            if self.use_cache:
                layer_past.conv_state = x[:, -3:, ...]

        x = x + self.conv1d(x=x)

        if has_layer_past:
            x = x[:, -1:, ...]

        zh = self.input_zh(x)

        z, h = torch.chunk(zh, chunks=2, dim=-1)

        z = z.sigmoid()

        cache = None
        if not has_layer_past:
            h_o = linear_scan((1 - z), (z * h))

            if self.use_cache:
                assert h_o is not None
                layer_past.update_cache(h_o[:, -1:, :])
                cache = layer_past

        else:
            h_o = ((1 - z) * layer_past.recc_state) + (z * h)

            layer_past.update_cache(h_o[:, -1:, :])
            cache = layer_past

        output = self.resid_proj(h_o)

        return output, cache


class RNNLayer(nn.Module):
    def __init__(self, config: GRUConfig, use_cache: bool = False):
        super().__init__()
        self.use_cache = use_cache

        self.recc = GRUBlock(config, use_cache)

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


class GRUModel(nn.Module):
    def __init__(self, config: GRUConfig, use_cache=False):
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

        self.lm_head.weight = self.embed_tokens.weight

    def reset_parameters(self):
        lecun_init(self.embed_tokens.weight, self.config.hidden_size)

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

        logits = self.lm_head(hidden_states)

        if self.use_cache:
            return logits, rnn_rnn_cache_list

        else:
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()

                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                return loss
            return logits
