import torch


class RNNCache:
    state: torch.Tensor
    device: torch.device
    current_cache_size: int = 0

    def __init__(self, state_dim: int, device: torch.device):
        self.recc_state = torch.full(
            (1, 1, state_dim),
            fill_value=torch.nan,
            dtype=torch.bfloat16,
            device=device,
        )

        self.conv_state = torch.full(
            (1, 3, state_dim),  # window_size - 1
            fill_value=torch.nan,
            dtype=torch.bfloat16,
            device=device,
        )

        self.state_dim = state_dim
        self.device = device

    def update_cache(self, new_state: torch.Tensor) -> None:
        assert new_state.shape[0] == 1
        assert new_state.shape[1] == 1
        assert new_state.ndim == 3

        self.recc_state[...] = new_state
        self.current_cache_size = 1

    def update_conv_cache(self, new_state: torch.Tensor) -> None:
        assert new_state.shape[0] == 1
        assert new_state.shape[1] == 1
        assert new_state.ndim == 3

        self.conv_state = torch.roll(self.conv_state, shifts=-1, dims=1)
        self.conv_state[:, -1, :] = new_state

    def __repr__(self) -> str:
        return f"RNNCache: current_cache_size={self.current_cache_size}, state_dim={self.state_dim}"
