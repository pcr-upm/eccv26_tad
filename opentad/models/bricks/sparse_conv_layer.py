import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
import math
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import opentad_sparse_ops as sparse_lib_opt
except ImportError:
    source_files_opt = [
        os.path.join(current_dir, "2d_sparse_v3_bind.cpp"),
        os.path.join(current_dir, "2d_sparse_v3.cu"),
        os.path.join(current_dir, "implicit_gemm_v2.cu"),
    ]

    sparse_lib_opt = load(
        name="sparse_fwd_opt_v6",
        sources=source_files_opt,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class SparseConv2dFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, x, neighbor_indices, weight, bias=None, tokens_per_batch=0, chunk_size=0
    ):
        # weight is pre-permuted: (9, C_in, C_out)
        _, C_in, C_out = weight.shape

        # Ensure C_in is divisible by 4 for the optimized kernel
        if C_in % 4 != 0:
            raise ValueError(
                f"C_in ({C_in}) must be divisible by 4 for optimized kernel."
            )

        w_opt = weight

        # Save for backward
        ctx.save_for_backward(x, neighbor_indices, w_opt, bias)
        ctx.C_in = C_in
        ctx.tokens_per_batch = tokens_per_batch

        # 1. Forward Convolution
        # Use Implicit GEMM V2 for channels >= 128.
        # V2 is faster than Custom at C=128 for small N, and competitive for large N.
        # For C >= 256, V2 is consistently superior or equal.
        if C_in >= 128:
            # Use chunked version if chunk_size > 0 for memory efficiency
            if chunk_size > 0:
                output = sparse_lib_opt.implicit_gemm_fwd_v2_chunked(
                    x,
                    neighbor_indices,
                    w_opt,
                    bias if bias is not None else torch.Tensor(),
                    tokens_per_batch,
                    chunk_size,
                )
            else:
                output = sparse_lib_opt.implicit_gemm_fwd_v2(
                    x,
                    neighbor_indices,
                    w_opt,
                    bias if bias is not None else torch.Tensor(),
                    tokens_per_batch,
                )
            # Implicit GEMM already adds bias
            return output
        else:
            output = sparse_lib_opt.fwd(
                x,
                neighbor_indices,
                w_opt,
                C_out,
                tokens_per_batch,
            )

            # 2. Add Bias
            if bias is not None:
                output += bias.view(1, -1)  # Broadcast add

            return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, neighbor_indices, w_opt, bias = ctx.saved_tensors
        C_in = ctx.C_in
        tokens_per_batch = ctx.tokens_per_batch

        grad_input = grad_weight = grad_bias = None

        # 1. Input Gradient
        if ctx.needs_input_grad[0]:
            grad_input = sparse_lib_opt.bwd_in(
                grad_output,
                neighbor_indices,
                w_opt,
                x.numel() // x.shape[-1],
                C_in,
                tokens_per_batch,
            )
            if x.dim() == 3:
                grad_input = grad_input.view(x.shape)

        # 2. Weight Gradient
        if ctx.needs_input_grad[2]:
            grad_weight = sparse_lib_opt.bwd_w(
                grad_output, x, neighbor_indices, C_in, tokens_per_batch
            )

        # 3. Bias Gradient (Sum over N dimensions)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)  # Sum across all pixels/batch

        return grad_input, None, grad_weight, grad_bias, None, None


class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, chunk_size=4096):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_size = (
            chunk_size  # 0 = no chunking, >0 = memory-efficient chunked processing
        )

        # Pre-permuted weight layout: (9, C_in, C_out) - avoids permute in forward pass
        self.weight = nn.Parameter(torch.Tensor(9, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Create temporary weight in standard format for initialization
        temp_weight = torch.empty(self.out_channels, self.in_channels, 3, 3)
        nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))

        # Transform to pre-permuted layout: (C_out, C_in, 3, 3) -> (C_out, C_in, 9) -> (9, C_in, C_out)
        with torch.no_grad():
            self.weight.copy_(
                temp_weight.view(self.out_channels, self.in_channels, 9).permute(
                    2, 1, 0
                )
            )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, neighbor_indices):
        # x: [N_kept, C_in] or [B, N, C_in]
        # neighbor_indices: [N_kept, 9] or [B, N, 9]

        # Handle batch dimension if present
        is_batched = x.dim() == 3
        tokens_per_batch = 0
        if is_batched:
            B, N, C = x.shape
            # Heuristic to detect if indices are local or global
            # If local, we need to tell the kernel to add batch offsets
            if neighbor_indices.numel() > 0:
                max_idx = neighbor_indices.max()
                if max_idx < N:
                    tokens_per_batch = N

            # Flatten x to (B*N, C)
            x = x.reshape(-1, C)

            neighbor_indices = neighbor_indices.reshape(-1, 9)

        # Ensure neighbor_indices is int32 for C++ kernel
        if neighbor_indices.dtype != torch.int32:
            neighbor_indices = neighbor_indices.to(torch.int32)

        weight = self.weight
        bias = self.bias

        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            if x.dtype != target_dtype:
                x = x.to(target_dtype)
            if weight.dtype != target_dtype:
                weight = weight.to(target_dtype)
            if bias is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
        elif x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
            if bias is not None:
                bias = bias.to(x.dtype)

        output = SparseConv2dFunction.apply(
            x, neighbor_indices, weight, bias, tokens_per_batch, self.chunk_size
        )

        if is_batched:
            output = output.view(B, N, -1)

        return output
