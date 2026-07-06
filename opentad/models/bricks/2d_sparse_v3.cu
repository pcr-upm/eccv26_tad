#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// =============================================================================
// Helper Functions & Math
// =============================================================================

// --- Float4 Math ---
__device__ __forceinline__ float4 add_f4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 mul_scalar_f4(float s, float4 v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

// --- Loading and Unpacking ---
// Generic template declaration to handle dispatch types
template<typename T>
__device__ __forceinline__ void load_and_unpack(const T* addr, float* out);

// 1. Float Specialization (Vectorized load 128-bit)
template<>
__device__ __forceinline__ void load_and_unpack<float>(const float* addr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(addr);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

// 2. Half Specialization (Packed load 64-bit -> float)
template<>
__device__ __forceinline__ void load_and_unpack<at::Half>(const at::Half* addr, float* out) {
    int2 v = *reinterpret_cast<const int2*>(addr); // Load 8 bytes (4 halves)
    const __half* h_ptr = reinterpret_cast<const __half*>(&v);
    out[0] = __half2float(h_ptr[0]);
    out[1] = __half2float(h_ptr[1]);
    out[2] = __half2float(h_ptr[2]);
    out[3] = __half2float(h_ptr[3]);
}

// 3. BFloat16 Specialization (Packed load 64-bit -> float)
template<>
__device__ __forceinline__ void load_and_unpack<at::BFloat16>(const at::BFloat16* addr, float* out) {
    int2 v = *reinterpret_cast<const int2*>(addr); // Load 8 bytes (4 bfloats)
    const __nv_bfloat16* b_ptr = reinterpret_cast<const __nv_bfloat16*>(&v);
    out[0] = __bfloat162float(b_ptr[0]);
    out[1] = __bfloat162float(b_ptr[1]);
    out[2] = __bfloat162float(b_ptr[2]);
    out[3] = __bfloat162float(b_ptr[3]);
}

// 4. Double Specialization (Scalar load fallback)
template<>
__device__ __forceinline__ void load_and_unpack<double>(const double* addr, float* out) {
    out[0] = static_cast<float>(addr[0]);
    out[1] = static_cast<float>(addr[1]);
    out[2] = static_cast<float>(addr[2]);
    out[3] = static_cast<float>(addr[3]);
}

// --- 8-Element Loading (Vectorized 256-bit for float, 128-bit for half/bf16) ---
template<typename T>
__device__ __forceinline__ void load_and_unpack_8(const T* addr, float* out);

// Float: 2x float4 loads
template<>
__device__ __forceinline__ void load_and_unpack_8<float>(const float* addr, float* out) {
    load_and_unpack<float>(addr, out);
    load_and_unpack<float>(addr + 4, out + 4);
}

// Half: 1x int4 load (16 bytes = 8 halves)
template<>
__device__ __forceinline__ void load_and_unpack_8<at::Half>(const at::Half* addr, float* out) {
    int4 v = *reinterpret_cast<const int4*>(addr);
    
    const __half* h;
    h = reinterpret_cast<const __half*>(&v.x); out[0] = __half2float(h[0]); out[1] = __half2float(h[1]);
    h = reinterpret_cast<const __half*>(&v.y); out[2] = __half2float(h[0]); out[3] = __half2float(h[1]);
    h = reinterpret_cast<const __half*>(&v.z); out[4] = __half2float(h[0]); out[5] = __half2float(h[1]);
    h = reinterpret_cast<const __half*>(&v.w); out[6] = __half2float(h[0]); out[7] = __half2float(h[1]);
}

// BFloat16: 1x int4 load (16 bytes = 8 bfloats)
template<>
__device__ __forceinline__ void load_and_unpack_8<at::BFloat16>(const at::BFloat16* addr, float* out) {
    int4 v = *reinterpret_cast<const int4*>(addr);

    const __nv_bfloat16* b;
    b = reinterpret_cast<const __nv_bfloat16*>(&v.x); out[0] = __bfloat162float(b[0]); out[1] = __bfloat162float(b[1]);
    b = reinterpret_cast<const __nv_bfloat16*>(&v.y); out[2] = __bfloat162float(b[0]); out[3] = __bfloat162float(b[1]);
    b = reinterpret_cast<const __nv_bfloat16*>(&v.z); out[4] = __bfloat162float(b[0]); out[5] = __bfloat162float(b[1]);
    b = reinterpret_cast<const __nv_bfloat16*>(&v.w); out[6] = __bfloat162float(b[0]); out[7] = __bfloat162float(b[1]);
}

template<>
__device__ __forceinline__ void load_and_unpack_8<double>(const double* addr, float* out) {
    load_and_unpack<double>(addr, out);
    load_and_unpack<double>(addr + 4, out + 4);
}

// Helper to load weight vector (always converts to float4 for accumulation)
template<typename T>
__device__ __forceinline__ float4 load_w_vec(const T* addr) {
    float vals[4];
    load_and_unpack<T>(addr, vals);
    return make_float4(vals[0], vals[1], vals[2], vals[3]);
}

// --- Packing and Storing ---
template<typename T>
__device__ __forceinline__ void pack_and_store(T* addr, float4 v);

template<>
__device__ __forceinline__ void pack_and_store<float>(float* addr, float4 v) {
    *reinterpret_cast<float4*>(addr) = v;
}

template<>
__device__ __forceinline__ void pack_and_store<at::Half>(at::Half* addr, float4 v) {
    __half2 h1 = __float22half2_rn(make_float2(v.x, v.y));
    __half2 h2 = __float22half2_rn(make_float2(v.z, v.w));
    int2 val;
    val.x = *reinterpret_cast<int*>(&h1);
    val.y = *reinterpret_cast<int*>(&h2);
    *reinterpret_cast<int2*>(addr) = val;
}

template<>
__device__ __forceinline__ void pack_and_store<at::BFloat16>(at::BFloat16* addr, float4 v) {
    __nv_bfloat16 b1 = __float2bfloat16(v.x);
    __nv_bfloat16 b2 = __float2bfloat16(v.y);
    __nv_bfloat16 b3 = __float2bfloat16(v.z);
    __nv_bfloat16 b4 = __float2bfloat16(v.w);
    
    unsigned short s1 = *reinterpret_cast<unsigned short*>(&b1);
    unsigned short s2 = *reinterpret_cast<unsigned short*>(&b2);
    unsigned short s3 = *reinterpret_cast<unsigned short*>(&b3);
    unsigned short s4 = *reinterpret_cast<unsigned short*>(&b4);
    
    int2 val;
    val.x = (int)s1 | ((int)s2 << 16);
    val.y = (int)s3 | ((int)s4 << 16);
    *reinterpret_cast<int2*>(addr) = val;
}

template<>
__device__ __forceinline__ void pack_and_store<double>(double* addr, float4 v) {
    addr[0] = static_cast<double>(v.x);
    addr[1] = static_cast<double>(v.y);
    addr[2] = static_cast<double>(v.z);
    addr[3] = static_cast<double>(v.w);
}

// --- Warp Reduce ---
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void gpuAtomicAdd(float* address, float val) {
    atomicAdd(address, val);
}

// =============================================================================
// 1. Forward Optimized Kernel (Thread Coarsening: 2 Rows per Thread)
// =============================================================================
// [Legacy Kernel kept for reference or fallback if needed]
template <typename scalar_t>
__global__ void __launch_bounds__(256) sparse_fwd_opt_vec4_row2(
    const scalar_t* __restrict__ x, 
    const int* __restrict__ neighbor_indices, 
    const scalar_t* __restrict__ weight, 
    scalar_t* __restrict__ output,
    int n_kept, int C_in, int C_out, int chunks_per_row, int tokens_per_batch) 
{
    int row0, row1;
    int co_chunk;
    int offset0 = 0, offset1 = 0;
    bool has_row1 = false;

    if (tokens_per_batch > 0) {
        // Batched Mode: Grid (BlocksX, BatchSize)
        int batch_idx = blockIdx.y;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        int row_pair = tid / chunks_per_row;
        co_chunk = tid % chunks_per_row;

        int row0_local = row_pair * 2;
        if (row0_local >= tokens_per_batch) return;

        row0 = batch_idx * tokens_per_batch + row0_local;
        offset0 = batch_idx * tokens_per_batch;

        int row1_local = row0_local + 1;
        if (row1_local < tokens_per_batch) {
            row1 = row0 + 1;
            offset1 = offset0;
            has_row1 = true;
        }
    } else {
        // Standard Mode: Grid (Blocks, 1)
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int row_pair = tid / chunks_per_row;
        co_chunk = tid % chunks_per_row;

        row0 = row_pair * 2;
        if (row0 >= n_kept) return;

        row1 = row0 + 1;
        has_row1 = (row1 < n_kept);
    }

    float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const int* neighbors0 = &neighbor_indices[row0 * 9];
    const int* neighbors1 = has_row1 ? &neighbor_indices[row1 * 9] : nullptr;
    
    // Weight layout: [9, C_in, C_out]
    int w_stride_K = C_in * C_out;    // Stride between k slices
    int w_base_offset_co = co_chunk * 4; 

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        int n_idx0 = neighbors0[k];
        int n_idx1 = has_row1 ? neighbors1[k] : -1;

        if (n_idx0 == -1 && n_idx1 == -1) continue;

        long w_k_base = k * w_stride_K + w_base_offset_co;

        // Unrolled Loop: Process 8 input channels at a time
        for (int ci = 0; ci < C_in; ci += 8) {
            float x0_vals[8] = {0.f};
            float x1_vals[8] = {0.f};

            // Load X for row 0 (8 elements)
            if (n_idx0 != -1) {
                long global_idx0 = (long)(n_idx0 + offset0);
                load_and_unpack_8<scalar_t>(&x[global_idx0 * C_in + ci], x0_vals);
            }
            // Load X for row 1 (8 elements)
            if (n_idx1 != -1) {
                long global_idx1 = (long)(n_idx1 + offset1);
                load_and_unpack_8<scalar_t>(&x[global_idx1 * C_in + ci], x1_vals);
            }

            // Load Weights and Accumulate
            // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float4 w = load_w_vec<scalar_t>(&weight[w_k_base + (ci + j) * C_out]);
                sum0 = add_f4(sum0, mul_scalar_f4(x0_vals[j], w));
                sum1 = add_f4(sum1, mul_scalar_f4(x1_vals[j], w));
            }
        }
    }

    // Write Output Row 0
    int co_start = co_chunk * 4;
    if (co_start + 4 <= C_out) {
        pack_and_store<scalar_t>(&output[(long)row0 * C_out + co_start], sum0);
    } else {
        for (int i = 0; i < (C_out - co_start); ++i) {
            float v = (i==0)?sum0.x : (i==1)?sum0.y : (i==2)?sum0.z : sum0.w;
            output[(long)row0 * C_out + co_start + i] = static_cast<scalar_t>(v);
        }
    }

    // Write Output Row 1
    if (has_row1) {
        if (co_start + 4 <= C_out) {
            pack_and_store<scalar_t>(&output[(long)row1 * C_out + co_start], sum1);
        } else {
            for (int i = 0; i < (C_out - co_start); ++i) {
                float v = (i==0)?sum1.x : (i==1)?sum1.y : (i==2)?sum1.z : sum1.w;
                output[(long)row1 * C_out + co_start + i] = static_cast<scalar_t>(v);
            }
        }
    }
}

// =============================================================================
// 1b. Forward Tiled Kernel (Shared Memory for Weights & Inputs)
// =============================================================================
template <typename scalar_t>
__global__ void __launch_bounds__(256) sparse_fwd_tiled(
    const scalar_t* __restrict__ x,
    const int* __restrict__ neighbor_indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int n_kept, int C_in, int C_out, int tokens_per_batch)
{
    // Block: 32 Rows (y), 32 Channels (x) -> 8 chunks of 4
    // Threads: 256.
    // tid maps to (row_in_block, co_chunk_in_block)
    // row_in_block: tid / 8  (0..31)
    // co_chunk_in_block: tid % 8 (0..7)
    
    int tid = threadIdx.x;
    int row_in_block = tid / 8;
    int co_chunk_in_block = tid % 8;
    
    int block_row_start = blockIdx.y * 32;
    int block_co_chunk_start = blockIdx.x * 8;
    
    int global_row = block_row_start + row_in_block;
    int global_co_chunk = block_co_chunk_start + co_chunk_in_block;
    
    // Shared Memory
    // smem_x: [32 rows][8 scalars]
    // smem_w: [8 ci][8 co_chunks] (float4)
    __shared__ float smem_x[32][8];
    __shared__ float4 smem_w[8][8];
    
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    
    int offset = 0;
    if (tokens_per_batch > 0) {
        int batch_idx = global_row / tokens_per_batch;
        offset = batch_idx * tokens_per_batch;
    }
    
    // Loop over neighbors
    for (int k = 0; k < 9; ++k) {
        
        int n_idx = -1;
        if (global_row < n_kept) {
            n_idx = neighbor_indices[global_row * 9 + k];
        }
        
        // Loop over C_in chunks
        for (int ci_base = 0; ci_base < C_in; ci_base += 8) {
            
            __syncthreads();
            
            // 1. Load Weights (Threads 0..63)
            if (tid < 64) {
                int w_ci = tid / 8; // 0..7
                int w_co = tid % 8; // 0..7
                
                int global_w_ci = ci_base + w_ci;
                int global_w_co_chunk = block_co_chunk_start + w_co;
                
                if (global_w_ci < C_in && (global_w_co_chunk * 4) < C_out) {
                     // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
                     long w_idx = (long)k * C_in * C_out + global_w_ci * C_out + global_w_co_chunk * 4;
                     smem_w[w_ci][w_co] = load_w_vec<scalar_t>(&weight[w_idx]);
                } else {
                     smem_w[w_ci][w_co] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            
            // 2. Load Inputs (Threads with co_chunk_in_block == 0 -> 0, 8, 16... 248)
            if (co_chunk_in_block == 0) {
                float vals[8] = {0.f};
                if (n_idx != -1) {
                    long global_input_idx = (long)(n_idx + offset);
                    load_and_unpack_8<scalar_t>(&x[global_input_idx * C_in + ci_base], vals);
                }
                #pragma unroll
                for (int i=0; i<8; ++i) smem_x[row_in_block][i] = vals[i];
            }
            
            __syncthreads();
            
            // 3. Compute
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float val_x = smem_x[row_in_block][j];
                float4 val_w = smem_w[j][co_chunk_in_block];
                sum = add_f4(sum, mul_scalar_f4(val_x, val_w));
            }
        }
    }
    
    // Store Output
    if (global_row < n_kept && (global_co_chunk * 4) < C_out) {
        long out_idx = (long)global_row * C_out + global_co_chunk * 4;
        if ((global_co_chunk * 4 + 4) <= C_out) {
             pack_and_store<scalar_t>(&output[out_idx], sum);
        } else {
             for (int i = 0; i < (C_out - global_co_chunk * 4); ++i) {
                float v = (i==0)?sum.x : (i==1)?sum.y : (i==2)?sum.z : sum.w;
                output[out_idx + i] = static_cast<scalar_t>(v);
            }
        }
    }
}

// =============================================================================
// 1c. Forward Tiled Kernel V2 (Optimized Loading & Compute Mapping)
// =============================================================================
// TILE_M (Rows) = 32
// TILE_N (C_out) = 32 (8 chunks of float4)
// TILE_K (C_in) = 32 (Increased to 32 to improve coalescing and cache line usage)
// Threads = 128
template <typename scalar_t>
__global__ void __launch_bounds__(128) sparse_fwd_tiled_v2(
    const scalar_t* __restrict__ x,
    const int* __restrict__ neighbor_indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int n_kept, int C_in, int C_out, int tokens_per_batch)
{
    // Thread Mapping
    int tid = threadIdx.x;
    
    // Compute Mapping: Each thread computes 2 rows x 1 chunk (4 channels)
    // Grid: (C_out / 32, N_kept / 32)
    int block_row_start = blockIdx.y * 32;
    int block_co_chunk_start = blockIdx.x * 8; // 8 chunks = 32 channels
    
    // Thread coordinates for compute
    int ty = tid / 8; // 0..15
    int tx = tid % 8; // 0..7
    
    // Shared Memory
    // smem_x: [32 rows][32 channels] -> [32][33] floats (Padded to avoid bank conflicts)
    // smem_w: [32 inputs][8 chunks] -> [32][9] float4s (Padded to avoid bank conflicts)
    __shared__ float smem_x[32][33];
    __shared__ float4 smem_w[32][9];
    
    // Accumulators
    float4 sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 sum1 = make_float4(0.f, 0.f, 0.f, 0.f);
    
    int offset = 0;
    if (tokens_per_batch > 0) {
        int batch_idx = block_row_start / tokens_per_batch;
        offset = batch_idx * tokens_per_batch;
    }
    
    // Loop over neighbors
    for (int k = 0; k < 9; ++k) {
        
        // Loop over C_in in blocks of 32 (TILE_K)
        for (int ci_base = 0; ci_base < C_in; ci_base += 32) {
            
            // --- Load Phase ---
            __syncthreads();
            
            // 1. Load Weights: 32 * 8 float4s = 256 items. 128 threads.
            // Each thread loads 2 float4s.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int load_idx = tid + i * 128; // 0..255
                int w_row = load_idx / 8; // 0..31 (matches TILE_K)
                int w_col = load_idx % 8; // 0..7 (matches TILE_N chunks)
                
                int global_w_ci = ci_base + w_row;
                int global_w_co_chunk = block_co_chunk_start + w_col;
                
                if (global_w_ci < C_in && (global_w_co_chunk * 4) < C_out) {
                    // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
                    long w_idx = (long)k * C_in * C_out + global_w_ci * C_out + global_w_co_chunk * 4;
                    smem_w[w_row][w_col] = load_w_vec<scalar_t>(&weight[w_idx]);
                } else {
                    smem_w[w_row][w_col] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            
            // 2. Load Inputs: 32 rows * 32 channels.
            // We load as float4s: 32 * 8 float4s = 256 float4s.
            // 128 threads. Each thread loads 2 float4s.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int load_idx = tid + i * 128; // 0..255
                // Map load_idx to (row, col_chunk_of_4)
                int row = load_idx / 8; // 0..31
                int col_chunk = load_idx % 8; // 0..7
                
                int global_row = block_row_start + row;
                int global_ci = ci_base + col_chunk * 4;
                
                float vals[4] = {0.f, 0.f, 0.f, 0.f};
                
                if (global_row < n_kept) {
                    int n_idx = neighbor_indices[global_row * 9 + k];
                    if (n_idx != -1) {
                        long global_input_idx = (long)(n_idx + offset);
                        if (global_ci < C_in) {
                             load_and_unpack<scalar_t>(&x[global_input_idx * C_in + global_ci], vals);
                        }
                    }
                }
                
                smem_x[row][col_chunk * 4 + 0] = vals[0];
                smem_x[row][col_chunk * 4 + 1] = vals[1];
                smem_x[row][col_chunk * 4 + 2] = vals[2];
                smem_x[row][col_chunk * 4 + 3] = vals[3];
            }
            
            __syncthreads();
            
            // --- Compute Phase ---
            // Each thread computes 2 rows (ty*2 .. ty*2+1) for chunk tx
            #pragma unroll
            for (int kk = 0; kk < 32; ++kk) {
                float4 w_val = smem_w[kk][tx];
                
                float x_val0 = smem_x[ty * 2 + 0][kk];
                float x_val1 = smem_x[ty * 2 + 1][kk];
                
                sum0 = add_f4(sum0, mul_scalar_f4(x_val0, w_val));
                sum1 = add_f4(sum1, mul_scalar_f4(x_val1, w_val));
            }
        }
    }
    
    // Store Output
    int global_co_chunk = block_co_chunk_start + tx;
    int base_row = block_row_start + ty * 2;
    
    auto store_func = [&](int r_offset, float4 val) {
        int r = base_row + r_offset;
        if (r < n_kept && (global_co_chunk * 4) < C_out) {
            long out_idx = (long)r * C_out + global_co_chunk * 4;
            if ((global_co_chunk * 4 + 4) <= C_out) {
                pack_and_store<scalar_t>(&output[out_idx], val);
            } else {
                for (int i = 0; i < (C_out - global_co_chunk * 4); ++i) {
                    float v = (i==0)?val.x : (i==1)?val.y : (i==2)?val.z : val.w;
                    output[out_idx + i] = static_cast<scalar_t>(v);
                }
            }
        }
    };
    
    store_func(0, sum0);
    store_func(1, sum1);
}

// =============================================================================
// 1d. Forward Tiled Kernel V3 (Larger Tile N=64, 256 threads)
// =============================================================================
// TILE_M (Rows) = 32
// TILE_N (C_out) = 64 (16 chunks of float4)
// TILE_K (C_in) = 32
// Threads = 256
template <typename scalar_t>
__global__ void __launch_bounds__(256) sparse_fwd_tiled_v3(
    const scalar_t* __restrict__ x,
    const int* __restrict__ neighbor_indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int n_kept, int C_in, int C_out, int tokens_per_batch)
{
    int tid = threadIdx.x;
    int block_row_start = blockIdx.y * 32;
    int block_co_chunk_start = blockIdx.x * 16; 
    
    // Map 256 threads to 32 rows x 16 chunks.
    // Each thread computes 2 rows x 1 chunk.
    int ty = tid / 16; // 0..15
    int tx = tid % 16; // 0..15
    
    __shared__ float smem_x[32][33];
    __shared__ float4 smem_w[32][17];
    
    float4 sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 sum1 = make_float4(0.f, 0.f, 0.f, 0.f);
    
    int offset = 0;
    if (tokens_per_batch > 0) {
        int batch_idx = block_row_start / tokens_per_batch;
        offset = batch_idx * tokens_per_batch;
    }
    
    for (int k = 0; k < 9; ++k) {
        for (int ci_base = 0; ci_base < C_in; ci_base += 32) {
            __syncthreads();
            
            // 1. Load Weights: 32 * 16 = 512 float4s. 256 threads -> 2 per thread.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int load_idx = tid + i * 256; // 0..511
                int w_row = load_idx / 16;
                int w_col = load_idx % 16;
                int global_w_ci = ci_base + w_row;
                int global_w_co_chunk = block_co_chunk_start + w_col;
                if (global_w_ci < C_in && (global_w_co_chunk * 4) < C_out) {
                    // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
                    long w_idx = (long)k * C_in * C_out + global_w_ci * C_out + global_w_co_chunk * 4;
                    smem_w[w_row][w_col] = load_w_vec<scalar_t>(&weight[w_idx]);
                } else {
                    smem_w[w_row][w_col] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            
            // 2. Load Inputs: 32 rows * 32 channels = 256 float4s. 256 threads -> 1 per thread.
            int load_idx = tid;
            int row = load_idx / 8; // 0..31
            int col_chunk = load_idx % 8; // 0..7
            
            int global_row = block_row_start + row;
            int global_ci = ci_base + col_chunk * 4;
            float vals[4] = {0.f, 0.f, 0.f, 0.f};
            if (global_row < n_kept) {
                int n_idx = neighbor_indices[global_row * 9 + k];
                if (n_idx != -1) {
                    long global_input_idx = (long)(n_idx + offset);
                    if (global_ci < C_in) {
                         load_and_unpack<scalar_t>(&x[global_input_idx * C_in + global_ci], vals);
                    }
                }
            }
            smem_x[row][col_chunk * 4 + 0] = vals[0];
            smem_x[row][col_chunk * 4 + 1] = vals[1];
            smem_x[row][col_chunk * 4 + 2] = vals[2];
            smem_x[row][col_chunk * 4 + 3] = vals[3];
            
            __syncthreads();
            
            // Compute
            #pragma unroll
            for (int kk = 0; kk < 32; ++kk) {
                float x_val0 = smem_x[ty * 2 + 0][kk];
                float x_val1 = smem_x[ty * 2 + 1][kk];
                float4 w_val = smem_w[kk][tx];
                sum0 = add_f4(sum0, mul_scalar_f4(x_val0, w_val));
                sum1 = add_f4(sum1, mul_scalar_f4(x_val1, w_val));
            }
        }
    }
    
    int global_co_chunk = block_co_chunk_start + tx;
    int base_row = block_row_start + ty * 2;
    
    auto store_func = [&](int r_offset, float4 val) {
        int r = base_row + r_offset;
        if (r < n_kept && (global_co_chunk * 4) < C_out) {
            long out_idx = (long)r * C_out + global_co_chunk * 4;
            if ((global_co_chunk * 4 + 4) <= C_out) {
                pack_and_store<scalar_t>(&output[out_idx], val);
            } else {
                for (int i = 0; i < (C_out - global_co_chunk * 4); ++i) {
                    float v = (i==0)?val.x : (i==1)?val.y : (i==2)?val.z : val.w;
                    output[out_idx + i] = static_cast<scalar_t>(v);
                }
            }
        }
    };
    
    store_func(0, sum0);
    store_func(1, sum1);
}

// =============================================================================
// 2. Backward Input Kernel
// =============================================================================
template <typename scalar_t>
__global__ void sparse_bwd_in_vec4(
    const scalar_t* __restrict__ grad_output,
    const int* __restrict__ neighbor_indices,
    const scalar_t* __restrict__ weight,        // [9, C_in, C_out] (pre-permuted layout)
    float* __restrict__ grad_input,
    int n_kept, int C_in, int C_out, int tokens_per_batch)
{
    int warp_id = threadIdx.y; 
    int lane_id = threadIdx.x; 
    int row = blockIdx.x * blockDim.y + warp_id;

    if (row >= n_kept) return;

    const int* my_neighbors = &neighbor_indices[row * 9];
    const scalar_t* go_ptr = grad_output;
    const scalar_t* w_ptr = weight;

    int offset = 0;
    if (tokens_per_batch > 0) {
        offset = (row / tokens_per_batch) * tokens_per_batch;
    }

    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < 9; ++k) {
            int n_idx = my_neighbors[k];
            if (n_idx == -1) continue;

            float partial_sum = 0.0f;
            
            for (int co_chunk = lane_id; (co_chunk * 4) < C_out; co_chunk += 32) {
                long offset_go = (long)row * C_out + (co_chunk * 4);
                float vals[4];
                load_and_unpack<scalar_t>(&go_ptr[offset_go], vals);
                float4 g_vec = make_float4(vals[0], vals[1], vals[2], vals[3]);
                
                // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
                long w_offset = (long)k * C_in * C_out + ci * C_out + (co_chunk * 4);
                load_and_unpack<scalar_t>(&w_ptr[w_offset], vals);
                float4 w_vec = make_float4(vals[0], vals[1], vals[2], vals[3]);
                
                partial_sum += (g_vec.x * w_vec.x + g_vec.y * w_vec.y + 
                                g_vec.z * w_vec.z + g_vec.w * w_vec.w);
            }
            
            float total_update = warpReduceSum(partial_sum);

            if (lane_id == 0) {
                long global_n_idx = (long)(n_idx + offset);
                gpuAtomicAdd(&grad_input[global_n_idx * C_in + ci], total_update);
            }
        }
    }
}

// =============================================================================
// 3. Backward Weight Kernel
// =============================================================================
template <typename scalar_t>
__global__ void sparse_bwd_w_scalar(
    const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ x,
    const int* __restrict__ neighbor_indices, float* __restrict__ grad_weight,  // [9, C_in, C_out]
    int n_kept, int C_in, int C_out, int tokens_per_batch)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_kept || co >= C_out) return;

    float g_out = static_cast<float>(grad_output[(long)row * C_out + co]);
    const int* my_neighbors = &neighbor_indices[row * 9];
    int w_stride_K = C_in * C_out;  // Stride between k slices

    int offset = 0;
    if (tokens_per_batch > 0) {
        offset = (row / tokens_per_batch) * tokens_per_batch;
    }

    for (int k = 0; k < 9; ++k) {
        int n_idx = my_neighbors[k];
        if (n_idx == -1) continue;
        long global_n_idx = (long)(n_idx + offset);
        long x_base = global_n_idx * C_in;
        // Weight layout: [9, C_in, C_out] -> offset = k * C_in * C_out + ci * C_out + co
        long gw_base = k * w_stride_K + co;
        for (int ci = 0; ci < C_in; ++ci) {
            float input_val = static_cast<float>(x[x_base + ci]);
            gpuAtomicAdd(&grad_weight[gw_base + ci * C_out], g_out * input_val);
        }
    }
}

// =============================================================================
// Launchers
// =============================================================================

torch::Tensor launch_fwd(torch::Tensor x, torch::Tensor n_idx, torch::Tensor w, int C_out, int tokens_per_batch) {
    int n_kept = x.numel() / x.size(-1); // Total tokens (B*N)
    int C_in = x.size(-1);
    
    TORCH_CHECK(C_in % 8 == 0, "C_in must be divisible by 8 for this optimized kernel");

    // Output shape will be flattened (Total, C_out) initially
    auto out = torch::zeros({n_kept, C_out}, x.options());
    
    bool use_old_kernel = false;
    if (use_old_kernel) {
        int chunks_per_row = (C_out + 3) / 4;
        int rows_per_thread = 2;
        
        if (tokens_per_batch > 0) {
            int B = n_kept / tokens_per_batch;
            int N = tokens_per_batch;
            int tasks_per_batch = ((N + rows_per_thread - 1) / rows_per_thread) * chunks_per_row;
            int threads_per_block = 256;
            int blocks_x = (tasks_per_batch + threads_per_block - 1) / threads_per_block;
            dim3 grid(blocks_x, B);
            
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "sparse_fwd_opt_vec4_row2", ([&] {
                sparse_fwd_opt_vec4_row2<scalar_t><<<grid, threads_per_block>>>(
                    x.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), w.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 
                    n_kept, C_in, C_out, chunks_per_row, tokens_per_batch);
            }));
        } else {
            int total_threads = ((n_kept + rows_per_thread - 1) / rows_per_thread) * chunks_per_row;
            int threads_per_block = 256;
            int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "sparse_fwd_opt_vec4_row2", ([&] {
                sparse_fwd_opt_vec4_row2<scalar_t><<<blocks, threads_per_block>>>(
                    x.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), w.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 
                    n_kept, C_in, C_out, chunks_per_row, tokens_per_batch);
            }));
        }
    } else {
        if (C_out >= 64) {
             int BLOCK_ROWS = 32;
             int grid_y = (n_kept + BLOCK_ROWS - 1) / BLOCK_ROWS;
             int grid_x = (C_out + 63) / 64;
             dim3 grid(grid_x, grid_y);
             int threads_per_block = 256;
             AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "sparse_fwd_tiled_v3", ([&] {
                sparse_fwd_tiled_v3<scalar_t><<<grid, threads_per_block>>>(
                    x.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), w.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 
                    n_kept, C_in, C_out, tokens_per_batch);
            }));
        } else {
            // New Tiled Kernel V2 Configuration
            // Block: 32 Rows x 32 Channels (8 chunks)
            // Threads: 128
            
            int BLOCK_ROWS = 32;
            int grid_y = (n_kept + BLOCK_ROWS - 1) / BLOCK_ROWS;
            int grid_x = (C_out + 31) / 32; 
            
            dim3 grid(grid_x, grid_y);
            int threads_per_block = 128;
            
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "sparse_fwd_tiled_v2", ([&] {
                sparse_fwd_tiled_v2<scalar_t><<<grid, threads_per_block>>>(
                    x.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), w.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 
                    n_kept, C_in, C_out, tokens_per_batch);
            }));
        }
    }
    
    return out;
}

torch::Tensor launch_bwd_in(torch::Tensor go, torch::Tensor n_idx, torch::Tensor w, int n_kept, int C_in, int tokens_per_batch) {
    auto gi = torch::zeros({n_kept, C_in}, go.options().dtype(torch::kFloat32));
    dim3 block(32, 8);
    int rows_per_block = 8;
    int grid_x = (n_kept + rows_per_block - 1) / rows_per_block;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, go.scalar_type(), "sparse_bwd_in_vec4", ([&] {
        sparse_bwd_in_vec4<scalar_t><<<grid_x, block>>>(
            go.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), w.data_ptr<scalar_t>(), gi.data_ptr<float>(), n_kept, C_in, go.size(1), tokens_per_batch);
    }));
    return gi.to(go.scalar_type());
}

torch::Tensor launch_bwd_w(torch::Tensor go, torch::Tensor x, torch::Tensor n_idx, int C_in, int tokens_per_batch) {
    int C_out = go.size(1); 
    int n_kept = x.numel() / x.size(-1);
    // Output grad_weight in pre-permuted layout: [9, C_in, C_out]
    auto gw = torch::zeros({9, C_in, C_out}, go.options().dtype(torch::kFloat32));
    dim3 block(32, 8);
    dim3 grid((C_out + block.x - 1) / block.x, (n_kept + block.y - 1) / block.y);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, go.scalar_type(), "sparse_bwd_w_scalar", ([&] {
        sparse_bwd_w_scalar<scalar_t><<<grid, block>>>(
            go.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), n_idx.data_ptr<int>(), gw.data_ptr<float>(), n_kept, C_in, C_out, tokens_per_batch);
    }));
    return gw.to(x.scalar_type());
}
