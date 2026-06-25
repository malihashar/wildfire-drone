"""
ConvLSTM
--------
Convolutional LSTM implementation for spatiotemporal wildfire prediction.

Architecture:
    ConvLSTMCell  — single recurrent cell with convolutional gates
    ConvLSTM      — stacked multi-layer ConvLSTM encoder
    WildfireConvLSTM — full model with ConvLSTM encoder + prediction head
"""

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# ConvLSTM Cell
# ──────────────────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.

    Parameters
    ----------
    in_channels  : int   — number of input feature channels
    hidden_dim   : int   — number of hidden state channels
    kernel_size  : int   — convolution kernel size (square)
    bias         : bool  — whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # gates: input, forget, output, cell — concatenated into one conv
        self.conv = nn.Conv2d(
            in_channels  + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,                              # (B, C, H, W)
        state: tuple[Tensor, Tensor] | None,    # (h, c) or None
    ) -> tuple[Tensor, Tensor]:
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)     # (B, C+hidden, H, W)
        gates    = self.conv(combined)           # (B, 4*hidden, H, W)

        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ──────────────────────────────────────────────────────────────────────────────
# Stacked ConvLSTM Encoder
# ──────────────────────────────────────────────────────────────────────────────

class ConvLSTM(nn.Module):
    """
    Stacked ConvLSTM encoder.

    Processes a sequence (T, C, H, W) through num_layers ConvLSTM cells.
    Returns the final hidden state of the last layer.

    Parameters
    ----------
    in_channels  : int  — input channels per frame
    hidden_dims  : list[int]  — hidden channels per layer
    kernel_size  : int  — convolution kernel size (shared across layers)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: list[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_layers = len(hidden_dims)

        cells = []
        for i, hidden_dim in enumerate(hidden_dims):
            c_in = in_channels if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(c_in, hidden_dim, kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(
        self, x: Tensor                    # (B, T, C, H, W)
    ) -> Tensor:                           # (B, hidden_dims[-1], H, W)
        B, T, C, H, W = x.shape

        states: list[tuple[Tensor, Tensor] | None] = [None] * self.num_layers

        for t in range(T):
            frame = x[:, t]                # (B, C, H, W)
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(frame, states[layer_idx])
                states[layer_idx] = (h, c)
                frame = h                  # pass hidden state to next layer

        # return last layer's final hidden state
        assert states[-1] is not None
        return states[-1][0]              # (B, hidden_dims[-1], H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Full Prediction Model
# ──────────────────────────────────────────────────────────────────────────────

class WildfireConvLSTM(nn.Module):
    """
    End-to-end wildfire spread prediction model.

    Architecture:
        - Optional 2D input projection (Conv2d) applied per-frame
        - Stacked ConvLSTM encoder (default 2 layers × 64 hidden channels)
        - 1×1 Conv prediction head → sigmoid → fire probability map

    Parameters
    ----------
    in_channels  : int        — input channels (e.g. 10)
    hidden_dims  : list[int]  — hidden channels per ConvLSTM layer
    kernel_size  : int        — ConvLSTM kernel size
    proj_channels: int | None — if set, project input to this dim before ConvLSTM
    dropout      : float      — spatial dropout applied to encoder output
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dims: list[int] | None = None,
        kernel_size: int = 3,
        proj_channels: int | None = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # ── optional input projection ─────────────────────────────────────────
        if proj_channels is not None:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_channels),
                nn.ReLU(inplace=True),
            )
            encoder_in = proj_channels
        else:
            self.input_proj = nn.Identity()
            encoder_in = in_channels

        # ── ConvLSTM encoder ──────────────────────────────────────────────────
        self.encoder = ConvLSTM(encoder_in, hidden_dims, kernel_size)

        # ── prediction head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),   # → (B, 1, H, W)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (B, T, C, H, W)

        Returns
        -------
        Tensor  shape (B, H, W)  — fire probability in [0, 1]
        """
        B, T, C, H, W = x.shape

        # apply input projection frame-by-frame
        x_flat  = x.view(B * T, C, H, W)
        x_proj  = self.input_proj(x_flat)              # (B*T, proj_C, H, W)
        _, pC, pH, pW = x_proj.shape
        x_proj  = x_proj.view(B, T, pC, pH, pW)       # (B, T, proj_C, H, W)

        h_last  = self.encoder(x_proj)                 # (B, hidden, H, W)
        logits  = self.head(h_last).squeeze(1)         # (B, H, W)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, C, H, W = 2, 20, 10, 64, 64
    model = WildfireConvLSTM(in_channels=C)
    x     = torch.randn(B, T, C, H, W)
    out   = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")   # expected (2, 64, 64)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
