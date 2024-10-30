"""PyTorch Mamba model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .cache import RNNCache
from .scan_fused import fused_scan

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
            self.config.dt_rank, self.config.intermediate_size, bias=True
        )

        # init for dt
        dt_init_std = self.config.dt_rank**-0.5 * 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        self.dt_proj.weight._no_reinit = True  # type: ignore

        dt = torch.exp(
            torch.rand(self.config.intermediate_size)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True  # type: ignore

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
        self.conv1d = nn.Conv1d(
            in_channels=self.config.intermediate_size,
            out_channels=self.config.intermediate_size,
            bias=True,
            kernel_size=self.temporal_width,
            groups=self.config.intermediate_size,
            padding=self.temporal_width - 1,
        )

        self.scan_fn = fused_scan

        self.post_norm = RMSNorm(dim=self.config.intermediate_size)

    def _ssm(self, x):
        n = self.A_log.shape[1]

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.config.dt_rank, n, n], dim=-1)

        delta = self.dt_proj(delta)

        scan_out = self.scan_fn(
            A,
            delta.mT.contiguous(),
            B.mT.contiguous(),
            C.mT.contiguous(),
            x.mT.contiguous(),
        )
        y = scan_out.mT.contiguous()

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

        #TODO: Add activation checkpointing 
        
        cache = None

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        (x, res) = torch.chunk(x_and_res, chunks=2, dim=-1)

        (b, l, d) = x.shape

        x = rearrange(x, "b l d_in -> b d_in l")
        x = F.silu(self.conv1d(x)[:, :, :l])
        x = rearrange(x, "b d_in l -> b l d_in")

        y = self._ssm(x)

        y = y * F.silu(res)

        y = self.post_norm(y)

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

        for name, p in self.named_parameters():
            if name in ["resid_proj.weight"]:
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_hidden_layers)
                )

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
            else:
                nn.init.normal_(module.weight, std=0.02)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

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
