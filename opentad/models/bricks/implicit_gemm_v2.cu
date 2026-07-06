#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

// Simple dispatch macro to avoid internal PyTorch macro issues
#define DISPATCH_FLOAT_HALF_BFLOAT(TYPE, NAME, ...) \
  [&] { \
    const auto& the_type = TYPE; \
    at::ScalarType _st = ::detail::scalar_type(the_type); \
    switch (_st) { \
      case at::ScalarType::Double: { \
        using scalar_t = double; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::Float: { \
        using scalar_t = float; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::Half: { \
        using scalar_t = at::Half; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::BFloat16: { \
        using scalar_t = at::BFloat16; \
        return __VA_ARGS__(); \
      } \
      default: \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'"); \
    } \
  }()

// =============================================================================
// Kernels
// =============================================================================

// Kernel to gather x into x_gathered [9, N, Cin]
// Handles padding (neighbor_idx == -1) by filling with 0.
// Grid: (9 * N * Cin + 255) / 256
// Block: 256
template <typename scalar_t>
__global__ void gather_padded_kernel(
    const scalar_t* __restrict__ x,              // [N, Cin]
    const int* __restrict__ neighbor_indices,    // [N, 9]
    scalar_t* __restrict__ x_gathered,           // [9, N, Cin]
    int N,
    int Cin,
    int tokens_per_batch,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Map linear index to (k, i, c)
        // x_gathered is [9, N, Cin]
        int c = idx % Cin;
        int rem = idx / Cin;
        int i = rem % N;
        int k = rem / N;
        
        int n_idx = neighbor_indices[i * 9 + k];
        
        scalar_t val = static_cast<scalar_t>(0);
        
        if (n_idx != -1) {
            // Handle batch offset if needed
            if (tokens_per_batch > 0) {
                int batch_idx = i / tokens_per_batch;
                n_idx += batch_idx * tokens_per_batch;
            }
            
            val = x[n_idx * Cin + c];
        }
        
        x_gathered[idx] = val;
    }
}

// Chunked version of gather kernel - handles offset within batch
template <typename scalar_t>
__global__ void gather_padded_kernel_chunked(
    const scalar_t* __restrict__ x,              // [N_total, Cin] - full input
    const int* __restrict__ neighbor_indices,    // [chunk_n, 9] - chunk indices
    scalar_t* __restrict__ x_gathered,           // [9, chunk_n, Cin]
    int chunk_n,
    int Cin,
    int tokens_per_batch,
    int chunk_batch_offset,                      // Starting token index of this chunk
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Map linear index to (k, i, c)
        // x_gathered is [9, chunk_n, Cin]
        int c = idx % Cin;
        int rem = idx / Cin;
        int i = rem % chunk_n;  // Local index within chunk
        int k = rem / chunk_n;
        
        int n_idx = neighbor_indices[i * 9 + k];
        
        scalar_t val = static_cast<scalar_t>(0);
        
        if (n_idx != -1) {
            // Handle batch offset if needed
            if (tokens_per_batch > 0) {
                // Global token index = chunk_batch_offset + local index
                int global_i = chunk_batch_offset + i;
                int batch_idx = global_i / tokens_per_batch;
                n_idx += batch_idx * tokens_per_batch;
            }
            
            val = x[n_idx * Cin + c];
        }
        
        x_gathered[idx] = val;
    }
}

// =============================================================================
// Fused Reduction + Bias + Cast Kernel
// =============================================================================
// Computes: output[i, c] = sum(k=0..8, input[k, i, c]) + bias[c]
// Then casts to output dtype
// Grid: (N * Cout + 255) / 256
// Block: 256

template <typename output_t>
__global__ void fused_reduce_bias_cast_kernel(
    const float* __restrict__ input,      // [9, N, Cout] in FP32
    const float* __restrict__ bias,       // [Cout] or nullptr
    output_t* __restrict__ output,        // [N, Cout]
    int N,
    int Cout,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int c = idx % Cout;
        int i = idx / Cout;
        
        // Sum over batch dimension (k=0..8)
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 9; k++) {
            sum += input[k * N * Cout + i * Cout + c];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Cast and store
        output[idx] = static_cast<output_t>(sum);
    }
}

// Dispatch helper for output type
inline void launch_fused_reduce_bias_cast(
    torch::Tensor input,         // [9, N, Cout] FP32
    torch::Tensor bias,          // [Cout] FP32 or empty
    torch::Tensor output,        // [N, Cout] target dtype
    int N,
    int Cout)
{
    int total = N * Cout;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    auto out_dtype = output.scalar_type();
    if (out_dtype == at::ScalarType::Half) {
        fused_reduce_bias_cast_kernel<at::Half><<<grid, block>>>(
            input.data_ptr<float>(), bias_ptr,
            output.data_ptr<at::Half>(), N, Cout, total);
    } else if (out_dtype == at::ScalarType::BFloat16) {
        fused_reduce_bias_cast_kernel<at::BFloat16><<<grid, block>>>(
            input.data_ptr<float>(), bias_ptr,
            output.data_ptr<at::BFloat16>(), N, Cout, total);
    } else {
        fused_reduce_bias_cast_kernel<float><<<grid, block>>>(
            input.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), N, Cout, total);
    }
}

// =============================================================================
// C++ Implementation
// =============================================================================

torch::Tensor implicit_gemm_fwd_v2(
    torch::Tensor x,
    torch::Tensor neighbor_indices,
    torch::Tensor weight,
    torch::Tensor bias,
    int tokens_per_batch) 
{
    // x: [N, Cin]
    // neighbor_indices: [N, 9]
    // weight: [9, Cin, Cout] (pre-permuted)
    
    int N = x.size(0);
    int Cin = x.size(1);
    int Cout = weight.size(2);
    
    auto options = x.options();
    
    // 1. Gather x -> x_gathered [9, N, Cin]
    // We do this to create a contiguous layout for StridedBatchedGemm
    // and to handle the -1 indices (padding) without CPU sync.
    auto x_gathered = torch::empty({9, N, Cin}, options);
    
    int total_elements = 9 * N * Cin;
    dim3 block(256);
    dim3 grid((total_elements + 255) / 256);
    
    DISPATCH_FLOAT_HALF_BFLOAT(x.scalar_type(), "gather_padded_kernel", ([&] {
        gather_padded_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            neighbor_indices.data_ptr<int>(),
            x_gathered.data_ptr<scalar_t>(),
            N,
            Cin,
            tokens_per_batch,
            total_elements
        );
    }));
    
    // 2. Weight is already pre-permuted [9, Cin, Cout], just ensure contiguous
    auto weight_permuted = weight.contiguous();
    
    // 3. GEMM (Strided Batched)
    // We compute: out_features[k] = x_gathered[k] @ weight_permuted[k]
    // x_gathered[k]: [N, Cin]
    // weight_permuted[k]: [Cin, Cout]
    // out_features[k]: [N, Cout]
    
    // Determine accumulation type
    cudaDataType_t cuda_input_type;
    cudaDataType_t cuda_output_type;
    cudaDataType_t cuda_compute_type;
    
    if (x.scalar_type() == torch::kHalf) {
        cuda_input_type = CUDA_R_16F;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    } else if (x.scalar_type() == torch::kBFloat16) {
        cuda_input_type = CUDA_R_16BF;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    } else {
        // FP32
        cuda_input_type = CUDA_R_32F;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    }

    // Allocate output [9, N, Cout] in FP32 for accumulation
    auto out_features = torch::empty({9, N, Cout}, options.dtype(torch::kFloat32));
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Cout, N, Cin,
        &alpha,
        weight_permuted.data_ptr(), cuda_input_type, Cout,  // Matrix 1 (B)
        Cin * Cout,                                         // Stride 1
        x_gathered.data_ptr(), cuda_input_type, Cin,        // Matrix 2 (A)
        N * Cin,                                            // Stride 2
        &beta,
        out_features.data_ptr(), cuda_output_type, Cout,    // Result (C)
        N * Cout,                                           // Stride C
        9,                                                  // Batch Count
        cuda_compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // 4. Fused reduction + bias + cast
    // Convert bias to FP32 for the kernel if needed
    torch::Tensor bias_fp32;
    if (bias.defined()) {
        bias_fp32 = bias.to(torch::kFloat32);
    }
    
    auto out = torch::empty({N, Cout}, options);
    launch_fused_reduce_bias_cast(out_features, bias_fp32, out, N, Cout);
    
    return out;
}

// =============================================================================
// Chunked Implementation - Memory Efficient
// =============================================================================

torch::Tensor implicit_gemm_fwd_v2_chunked(
    torch::Tensor x,
    torch::Tensor neighbor_indices,
    torch::Tensor weight,
    torch::Tensor bias,
    int tokens_per_batch,
    int chunk_size)
{
    // x: [N, Cin]
    // neighbor_indices: [N, 9]
    // weight: [9, Cin, Cout] (pre-permuted)
    
    int N = x.size(0);
    int Cin = x.size(1);
    int Cout = weight.size(2);
    
    auto options = x.options();
    
    // Determine accumulation type
    cudaDataType_t cuda_input_type;
    cudaDataType_t cuda_output_type;
    cudaDataType_t cuda_compute_type;
    
    if (x.scalar_type() == torch::kHalf) {
        cuda_input_type = CUDA_R_16F;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    } else if (x.scalar_type() == torch::kBFloat16) {
        cuda_input_type = CUDA_R_16BF;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    } else {
        cuda_input_type = CUDA_R_32F;
        cuda_output_type = CUDA_R_32F;
        cuda_compute_type = CUDA_R_32F;
    }
    
    // Allocate final output [N, Cout] in target dtype
    auto out = torch::empty({N, Cout}, options);
    
    // Weight is already pre-permuted [9, Cin, Cout], just ensure contiguous
    auto weight_permuted = weight.contiguous();
    
    // Convert bias to FP32 once for the fused kernel
    torch::Tensor bias_fp32;
    if (bias.defined()) {
        bias_fp32 = bias.to(torch::kFloat32);
    }
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    float alpha = 1.0f;
    float beta_first = 0.0f;
    
    // Process in chunks
    for (int start = 0; start < N; start += chunk_size) {
        int end = std::min(start + chunk_size, N);
        int chunk_n = end - start;
        
        // Slice neighbor indices for this chunk (we use full x for gathering)
        auto indices_chunk = neighbor_indices.slice(0, start, end);      // [chunk_n, 9]
        
        // Allocate chunk buffers
        auto x_gathered = torch::empty({9, chunk_n, Cin}, options);      // [9, chunk_n, Cin]
        auto out_features = torch::empty({9, chunk_n, Cout}, options.dtype(torch::kFloat32)); // [9, chunk_n, Cout]
        
        // 1. Gather for this chunk
        int total_elements = 9 * chunk_n * Cin;
        dim3 block(256);
        dim3 grid((total_elements + 255) / 256);
        
        // Compute batch offset for this chunk
        int chunk_batch_offset = (tokens_per_batch > 0) ? start : 0;
        
        DISPATCH_FLOAT_HALF_BFLOAT(x.scalar_type(), "gather_padded_kernel_chunked", ([&] {
            gather_padded_kernel_chunked<scalar_t><<<grid, block>>>(
                x.data_ptr<scalar_t>(),              // Full x for gathering neighbors
                indices_chunk.data_ptr<int>(),
                x_gathered.data_ptr<scalar_t>(),
                chunk_n,
                Cin,
                tokens_per_batch,
                chunk_batch_offset,
                total_elements
            );
        }));
        
        // 2. Batched GEMM for this chunk
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Cout, chunk_n, Cin,
            &alpha,
            weight_permuted.data_ptr(), cuda_input_type, Cout,
            Cin * Cout,
            x_gathered.data_ptr(), cuda_input_type, Cin,
            chunk_n * Cin,
            &beta_first,
            out_features.data_ptr(), cuda_output_type, Cout,
            chunk_n * Cout,
            9,
            cuda_compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        
        // 3. Fused reduction + bias + cast for this chunk
        auto chunk_out = out.slice(0, start, end);  // View into output
        launch_fused_reduce_bias_cast(out_features, bias_fp32, chunk_out, chunk_n, Cout);
    }
    
    return out;
}
