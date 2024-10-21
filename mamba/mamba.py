"""PyTorch Mamba model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from .cache import RNNCache
from .external import Conv1D
from .scan import linear_scan

# ------
# Config
# ------


@dataclass
class MambaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int  # this is the expansion factor
    num_hidden_layers: int
    dt_rank: int


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


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig, use_cache: bool = False):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(
            self.config.hidden_size, 2 * self.config.intermediate_size
        )
        self.x_proj = nn.Linear(
            self.config.intermediate_size,
            self.config.dt_rank + self.config.state_size * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(
            self.config.dt_rank, self.config.intermediate_size, bias=False
        )

        A = repeat(
            torch.arange(1, self.config.state_size + 1),
            "n -> d n",
            d=self.config.intermediate_size,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.config.intermediate_size))
        self.out_proj = nn.Linear(
            self.config.intermediate_size, self.config.hidden_size, bias=False
        )

        self.resid_proj = nn.Linear(
            self.config.intermediate_size, config.hidden_size, bias=False
        )

        self.use_cache = use_cache

        self.temporal_width = 4
        self.conv1d = Conv1D(
            width=self.config.intermediate_size,
            temporal_width=self.temporal_width,
        )

    def _ssm(self, x):

        (d_in, n) = self.A_log.shape
        b, l, d = x.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.config.dt_rank, n, n], dim=-1)

        delta = torch.nn.functional.softplus(self.dt_proj(delta))

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, x, "b l d_in, b l n, b l d_in -> b l d_in n")

        deltaA = deltaA.reshape(b, l, -1)
        deltaB_u = deltaB_u.reshape(b, l, -1)

        scan_out = linear_scan(deltaA, deltaB_u)

        scan_out = rearrange(scan_out, "b l (d n) -> b l d n", d=d, n=n)

        y = einsum(scan_out, C, "b l d_in n, b l n -> b l d_in")

        y = y + x * D

        return y

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

        cache = None

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        (x, res) = torch.chunk(x_and_res, chunks=2, dim=-1)

        x = F.silu(self.conv1d(x=x))

        y = self._ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output, cache


class RNNLayer(nn.Module):
    def __init__(self, config: MambaConfig, use_cache: bool = False):
        super().__init__()
        self.use_cache = use_cache

        self.recc = Mamba(config, use_cache)

        self.input_norm = RMSNorm(config.hidden_size)

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

        return hidden_states, rnn_cache


class MambaModel(nn.Module):
    def __init__(self, config: MambaConfig, use_cache=False):
        super().__init__()
        self.config = config
        self.use_cache: bool = use_cache
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [(RNNLayer(config, use_cache)) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

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
