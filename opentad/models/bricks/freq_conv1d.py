import torch
import torch.nn as nn


class TemporalFreqConv1D(nn.Module):
    """1D temporal-frequency convolution on ``(B, C, T)`` features.

    This is the 1D temporal adaptation of the ``freq_conv`` adapter used inside
    the ViT backbone (see ``VisionTransformerSparseAdapterPOGUISE``). Channels are
    split into two halves:

    - a *temporal* branch: depthwise ``Conv1d`` over the time axis.
    - a *frequency* branch: multi-resolution ``rFFT`` along time, a depthwise
      ``Conv1d`` over the (real, imag) spectrum, then ``irfft`` back, averaged
      over the resolutions.

    The output is a residual delta gated by a per-channel ``gamma`` that is
    zero-initialized, so the module starts as an identity and can be dropped
    into an already-tuned network without destabilizing it.

    Args:
        channels: number of input/output channels ``C``.
        kernel_size: kernel size for both depthwise convs.
        freq_windows: explicit list of rFFT window sizes (in time steps). If
            ``None``, uses ``[T, T // 2, T // 4]`` at runtime. Only windows that
            evenly divide ``T`` are used.
        gamma_init: initial value of the residual gate (small -> near identity).
    """

    def __init__(self, channels, kernel_size=3, freq_windows=None, gamma_init=1e-4):
        super().__init__()
        self.Ct = channels // 2
        self.Cf = channels - self.Ct
        self.freq_windows = freq_windows
        pad = kernel_size // 2

        # Temporal branch: depthwise Conv1d over T
        self.temporal_conv = nn.Conv1d(
            self.Ct, self.Ct, kernel_size, padding=pad, groups=self.Ct
        )
        # Frequency branch: depthwise Conv1d over the (real, imag) parts of the
        # multi-resolution rFFT along the temporal axis
        self.freq_conv = nn.Conv1d(
            2 * self.Cf, 2 * self.Cf, kernel_size, padding=pad, groups=2 * self.Cf
        )
        nn.init.constant_(self.temporal_conv.bias, 0.0)
        nn.init.constant_(self.freq_conv.bias, 0.0)

        # Zero-init residual gate -> starts as a no-op
        self.gamma = nn.Parameter(gamma_init * torch.ones(channels, 1))

    def _windows_for(self, T):
        """Multi-resolution rFFT window sizes (in time steps) for a length T."""
        if self.freq_windows is not None:
            windows = list(self.freq_windows)
        else:
            windows = [w for w in (T, T // 2, T // 4) if w >= 1]
        # Only keep windows that evenly divide T (needed for the chunked rFFT)
        return [w for w in windows if w >= 1 and T % w == 0]

    def forward(self, x, mask=None):
        # x: (B, C, T), mask: (B, T) bool/float or None
        if mask is not None:
            x = x * mask.unsqueeze(1).float()  # zero padded frames before FFT

        B, C, T = x.shape
        xt, xf = x[:, : self.Ct], x[:, self.Ct :]
        orig_dtype = x.dtype

        # Depthwise Conv1d and FFT have no BFloat16/Half CUDA kernels, so run
        # both branches in float32 with autocast disabled, then cast back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Temporal branch
            xt = self.temporal_conv(xt.float()).to(orig_dtype)

            # Frequency branch: multi-resolution rFFT along T + depthwise Conv1d
            xf = xf.float()
            out_f = torch.zeros_like(xf)
            windows = self._windows_for(T)
            for win in windows:
                chunks = T // win
                xc = xf.reshape(B, self.Cf, chunks, win)
                X = torch.fft.rfft(xc, dim=-1)  # (B, Cf, chunks, Fr) complex
                X = torch.view_as_real(X)  # (B, Cf, chunks, Fr, 2)
                Fr = X.shape[-2]
                # -> (B*chunks, 2*Cf, Fr) for the depthwise conv
                X = X.permute(0, 2, 1, 4, 3).reshape(B * chunks, 2 * self.Cf, Fr)
                Y = self.freq_conv(X)
                Y = Y.reshape(B, chunks, self.Cf, 2, Fr).permute(0, 2, 1, 4, 3)
                Y = torch.complex(Y[..., 0], Y[..., 1])
                y = torch.fft.irfft(Y, n=win, dim=-1)  # (B, Cf, chunks, win)
                out_f = out_f + y.reshape(B, self.Cf, T)
            out_f = (out_f / max(len(windows), 1)).to(orig_dtype)

        out = torch.cat([xt, out_f], dim=1)  # (B, C, T)
        out = out * self.gamma  # gated residual delta

        if mask is not None:
            out = out * mask.unsqueeze(1).float()
        return out
