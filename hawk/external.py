"""This code hase been taken and modified from the following source:
https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/layers.py
"""


import math

import torch
import torch.nn as nn
from einops import rearrange


class Conv1D(nn.Module):
    """A 1D temporal convolution layer."""

    def __init__(
        self,
        width: int,
        temporal_width: int,
        w_init_variance_scale: float = 0.01,
    ):
        """Initializes the Conv1D.

        Args:
            width: The number of features for both inputs and outputs.
            temporal_width: The size of the temporal receptive field of the
            convolution. In other words, how much back in time the convolution can
            look to produce an output.
            w_init_variance_scale: A parameter that scales the variance of the
            initialization of the weights.
            device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
            dtype: What dtype to use for initialziation.
        """
        super().__init__()

        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale

        self.w = nn.Parameter(
            torch.empty([self.temporal_width, self.width], dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.empty([width], dtype=torch.float32))

        std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
        torch.nn.init.normal_(self.w, mean=0.0, std=std)
        torch.nn.init.zeros_(self.b)

    def doc_mask_forward(self, x):
        """Calls the Conv1D.

        Args:
            x: Sequence of input activations.
            segment_pos: Position of each token in the sequence.
            state: The state containing the previous `self.temporal_width-1` inputs
            This is set to `None` in training mode.

        Returns:
            The output of the convolution and the updated state.
        """

        # 1. Training mode:
        # - The full sequence length need to be output.
        prompt_len = 0
        output_len = x.shape[1]
        state_dtype = x.dtype

        # 3. Perform the convolution:
        # - Initialize an accumulator for the convolution output.
        convolution_output = 0.0

        # - We cannot look back by more than the total sequence length
        #   ("valid" convolution).
        temporal_width = min(self.temporal_width, prompt_len + output_len)

        # - The convolution is implemented as a manual loop so that we can
        #   incorporate the window masking further below.
        for temporal_shift in range(temporal_width):
            start_idx, end_idx = self._convolution_window_indices(
                prompt_len=prompt_len,
                shift_back=temporal_shift,
                output_len=output_len,
            )
            x_window = x[:, start_idx:end_idx]

            x_window = self._pad_window(x_window, output_len)

            w = self.w[self.temporal_width - temporal_shift - 1][None, None, :]

            convolution_output += x_window * w

        # - Add the bias of the convolution.
        convolution_output += self.b[None, None]

        return convolution_output

    def forward(self, x):
        return self.doc_mask_forward(x)

    def _convolution_window_indices(
        self,
        *,
        prompt_len: int,
        shift_back: int,
        output_len: int,
    ) -> tuple[int, int]:
        """Calculates the start and end indices for the convolution window.

        Args:
        prompt_len: Length of the prompt (zero in training mode).
        shift_back: By how much the window should be shifted backwards.
        output_len: Sequence length of the output (sequence length in training
        mode, one in decoding mode).

        Returns:
        start_idx: The starting index for the convolution window.
        end_idx: The ending index for the convolution window.
        """
        start_idx = max(prompt_len - shift_back, 0)
        end_idx = prompt_len + output_len - shift_back
        return start_idx, end_idx

    def _pad_window(
        self,
        window,
        output_len: int,
    ):
        """Left-pads the window if it is shorter than the output sequence length."""
        batch_size, window_len, width = window.shape
        padding_len = output_len - window_len
        padding = torch.zeros(
            (batch_size, padding_len, width),
            dtype=window.dtype,
            device=window.device,
        )
        return torch.concatenate([padding, window], dim=1)

    def _pad_state(
        self,
        state,
    ):
        """Left-pads the state if it is shorter than the temporal width."""
        b, state_seq_len, d = state.shape
        padding_len = self.temporal_width - state_seq_len - 1
        padding = torch.zeros(
            (b, padding_len, d),
            dtype=state.dtype,
            device=state.device,
        )
        return torch.concatenate([padding, state], dim=1)


def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU uniformly on a ring between min_rad and max_rad."""
    with torch.no_grad():
        tensor.uniform_(min_rad**2 + eps, max_rad**2 + eps)
        tensor.log_().mul_(0.5)

        if transform == "softplus":
            return tensor.neg_().exp_().sub_(1.0).log_()
        else:
            raise NotImplementedError()


_MAX_SQRT_GRADIENT = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(
        self,
        width: int,
        num_blocks: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the BlockDiagonalLinear.

        Args:
          width: The number of dimensions of the input and output.
          num_blocks: The number of diagonal blocks in the layer.
          w_init_variance_scale: A parameters that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        # Parameters.
        self.w = nn.Parameter(
            torch.empty(
                [self.num_blocks, self.block_width, self.block_width],
                device=device,
                dtype=dtype,
            )
        )
        self.b = nn.Parameter(
            torch.empty([self.num_blocks, self.block_width], device=device, dtype=dtype)
        )

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight `w` of the layer."""
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks.
        x = rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)
