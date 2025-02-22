%%writefile attention.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cooperative_groups.h>
#include <time.h>

namespace cg = cooperative_groups;

// Macro para verificação de erros CUDA
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Macro para verificação de erros CURAND
#define CURAND_CHECK(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "CURANDassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

// Kernel para realizar a operação de scale+shift
__global__ void scale_shift_kernel(float* input, float* output, int n, float min_val, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = min_val + input[idx] * scale;
    }
}

// Kernel para inicializar um vetor com um valor constante (usado para m)
__global__ void init_value_kernel(float* arr, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}

/***************************************************
 * Versões Vetorizadas (para embed_dim múltiplo de 4)
 ***************************************************/

// Kernel de forward – versão vetorizada
__global__ void forward_kernel_vectorized(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l,
    float* __restrict__ m,
    const int seq_len,
    const int embed_dim,
    const int total_col_blocks,
    const int total_row_blocks,
    const int block_size_col,
    const int block_size_row,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    
    int bx = blockIdx.x;  // batch
    int by = blockIdx.y;  // head
    int tx = threadIdx.x; // linha
    int ty = threadIdx.y; // coluna

    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset  = (bx * gridDim.y * seq_len) + (by * seq_len);
    
    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* S_tile = V_tile + block_size_col * embed_dim;
    
    const float eps = 1e-10f;
    
    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d += 4) {
                int idx = qkv_offset + row_idx * embed_dim + d;
                if (d + 4 <= embed_dim) {
                    float4 q = *(float4*)&Q[idx];
                    *(float4*)&Q_tile[tx * embed_dim + d] = q;
                }
            }
        }
        
        float mi = (row_idx < seq_len) ? m[lm_offset + row_idx] : -INFINITY;
        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;
        
        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    if (d + 4 <= embed_dim) {
                        *(float4*)&K_tile[ty * embed_dim + d] = *(float4*)&K[idx];
                        *(float4*)&V_tile[ty * embed_dim + d] = *(float4*)&V[idx];
                    }
                }
            }
            block.sync();
            
            float row_max = -INFINITY;
            float row_sum = 0.0f;
            
            if (tx < block_size_row && row_idx < seq_len) {
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[local_col * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + local_col] = sum;
                    row_max = fmaxf(row_max, sum);
                }
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float val = expf(S_tile[tx * block_size_col + local_col] - row_max);
                    S_tile[tx * block_size_col + local_col] = val;
                    row_sum += val;
                }
            }
            block.sync();
            
            float m_new = fmaxf(mi, row_max);
            float l_new = expf(mi - m_new) * li + expf(row_max - m_new) * row_sum;
            
            if (tx < block_size_row && row_idx < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                        float p = S_tile[tx * block_size_col + local_col];
                        float4 v = *(float4*)&V_tile[local_col * embed_dim + d];
                        acc.x += p * v.x;
                        acc.y += p * v.y;
                        acc.z += p * v.z;
                        acc.w += p * v.w;
                    }
                    
                    int out_idx = qkv_offset + row_idx * embed_dim + d;
                    float scale_factor = 1.0f / (l_new + eps);
                    float4 prev_o = *(float4*)&O[out_idx];
                    *(float4*)&O[out_idx] = make_float4(
                        scale_factor * (li * expf(mi - m_new) * prev_o.x + expf(row_max - m_new) * acc.x),
                        scale_factor * (li * expf(mi - m_new) * prev_o.y + expf(row_max - m_new) * acc.y),
                        scale_factor * (li * expf(mi - m_new) * prev_o.z + expf(row_max - m_new) * acc.z),
                        scale_factor * (li * expf(mi - m_new) * prev_o.w + expf(row_max - m_new) * acc.w)
                    );
                }
            }
            
            mi = m_new;
            li = l_new;
        }
        
        if (tx < block_size_row && row_idx < seq_len) {
            m[lm_offset + row_idx] = mi;
            l[lm_offset + row_idx] = li;
        }
        block.sync();
    }
}

// Kernel de backward – versão vetorizada
__global__ void backward_kernel_vectorized(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ S,
    const float* __restrict__ l,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int seq_len,
    const int embed_dim,
    const int total_col_blocks,
    const int total_row_blocks,
    const int block_size_col,
    const int block_size_row,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    
    int bx = blockIdx.x;  
    int by = blockIdx.y;  
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset  = (bx * gridDim.y * seq_len) + (by * seq_len);
    
    extern __shared__ float s[];
    float* Q_tile  = s;
    float* K_tile  = Q_tile + block_size_row * embed_dim;
    float* V_tile  = K_tile + block_size_col * embed_dim;
    float* dO_tile = V_tile + block_size_col * embed_dim;
    float* S_tile  = dO_tile + block_size_row * embed_dim;
    
    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d += 4) {
                int idx = qkv_offset + row_idx * embed_dim + d;
                if (d + 4 <= embed_dim) {
                    *(float4*)&Q_tile[tx * embed_dim + d] = *(float4*)&Q[idx];
                    *(float4*)&dO_tile[tx * embed_dim + d] = *(float4*)&dO[idx];
                }
            }
        }
        
        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;
        
        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    if (d + 4 <= embed_dim) {
                        *(float4*)&K_tile[ty * embed_dim + d] = *(float4*)&K[idx];
                        *(float4*)&V_tile[ty * embed_dim + d] = *(float4*)&V[idx];
                    }
                }
            }
            block.sync();
            
            float row_max = -INFINITY;
            if (tx < block_size_row && row_idx < seq_len) {
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[local_col * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + local_col] = sum;
                    row_max = fmaxf(row_max, sum);
                }
                
                float row_sum = 0.0f;
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float exp_val = expf(S_tile[tx * block_size_col + local_col] - row_max);
                    S_tile[tx * block_size_col + local_col] = exp_val;
                    row_sum += exp_val;
                }
                
                float sum_dS = 0.0f;
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float ds = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        ds += dO_tile[tx * embed_dim + d] * V_tile[local_col * embed_dim + d];
                    }
                    float p = S_tile[tx * block_size_col + local_col] / (li + 1e-10f);
                    ds *= scale;
                    sum_dS += ds * p;
                    S_tile[tx * block_size_col + local_col] = p * (ds - sum_dS);
                }
            }
            block.sync();
            
            if (tx < block_size_row && row_idx < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                        float ds = S_tile[tx * block_size_col + local_col];
                        float4 key_val = *(float4*)&K_tile[local_col * embed_dim + d];
                        acc.x += ds * key_val.x;
                        acc.y += ds * key_val.y;
                        acc.z += ds * key_val.z;
                        acc.w += ds * key_val.w;
                    }
                    int idx = qkv_offset + row_idx * embed_dim + d;
                    atomicAdd(&dQ[idx + 0], acc.x);
                    atomicAdd(&dQ[idx + 1], acc.y);
                    atomicAdd(&dQ[idx + 2], acc.z);
                    atomicAdd(&dQ[idx + 3], acc.w);
                }
            }
            
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    float4 dk_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float4 dv_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int r = 0; r < block_size_row && (row_block * block_size_row + r) < seq_len; r++) {
                        float ds = S_tile[r * block_size_col + ty];
                        float4 q = *(float4*)&Q_tile[r * embed_dim + d];
                        float4 do_val = *(float4*)&dO_tile[r * embed_dim + d];
                        float p = S_tile[r * block_size_col + ty] / (l[lm_offset + (row_block * block_size_row + r)] + 1e-10f);
                        
                        dk_acc.x += ds * q.x;
                        dk_acc.y += ds * q.y;
                        dk_acc.z += ds * q.z;
                        dk_acc.w += ds * q.w;
                        
                        dv_acc.x += p * do_val.x;
                        dv_acc.y += p * do_val.y;
                        dv_acc.z += p * do_val.z;
                        dv_acc.w += p * do_val.w;
                    }
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    atomicAdd(&dK[idx + 0], dk_acc.x);
                    atomicAdd(&dK[idx + 1], dk_acc.y);
                    atomicAdd(&dK[idx + 2], dk_acc.z);
                    atomicAdd(&dK[idx + 3], dk_acc.w);
                    
                    atomicAdd(&dV[idx + 0], dv_acc.x);
                    atomicAdd(&dV[idx + 1], dv_acc.y);
                    atomicAdd(&dV[idx + 2], dv_acc.z);
                    atomicAdd(&dV[idx + 3], dv_acc.w);
                }
            }
            block.sync();
        }
    }
}

/***************************************************
 * Versões Escalares (para embed_dim não múltiplo de 4)
 ***************************************************/

// Kernel de forward – versão escalar
__global__ void forward_kernel_scalar(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l,
    float* __restrict__ m,
    const int seq_len,
    const int embed_dim,
    const int total_col_blocks,
    const int total_row_blocks,
    const int block_size_col,
    const int block_size_row,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    
    int bx = blockIdx.x;  
    int by = blockIdx.y;  
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset  = (bx * gridDim.y * seq_len) + (by * seq_len);

    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* S_tile = V_tile + block_size_col * embed_dim;

    const float eps = 1e-10f;

    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d++) {
                int idx = qkv_offset + row_idx * embed_dim + d;
                Q_tile[tx * embed_dim + d] = Q[idx];
            }
        }

        float mi = (row_idx < seq_len) ? m[lm_offset + row_idx] : -INFINITY;
        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;

        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    K_tile[ty * embed_dim + d] = K[idx];
                    V_tile[ty * embed_dim + d] = V[idx];
                }
            }
            block.sync();

            float row_max = -INFINITY;
            float row_sum = 0.0f;
            if (tx < block_size_row && row_idx < seq_len) {
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[local_col * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + local_col] = sum;
                    row_max = fmaxf(row_max, sum);
                }
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float val = expf(S_tile[tx * block_size_col + local_col] - row_max);
                    S_tile[tx * block_size_col + local_col] = val;
                    row_sum += val;
                }
            }
            block.sync();

            float m_new = fmaxf(mi, row_max);
            float l_new = expf(mi - m_new) * li + expf(row_max - m_new) * row_sum;

            if (tx < block_size_row && row_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    float acc = 0.0f;
                    for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                        float p = S_tile[tx * block_size_col + local_col];
                        acc += p * V_tile[local_col * embed_dim + d];
                    }
                    int out_idx = qkv_offset + row_idx * embed_dim + d;
                    float scale_factor = 1.0f / (l_new + eps);
                    O[out_idx] = scale_factor * (li * expf(mi - m_new) * O[out_idx] + expf(row_max - m_new) * acc);
                }
            }
            mi = m_new;
            li = l_new;
        }
        if (tx < block_size_row && row_idx < seq_len) {
            m[lm_offset + row_idx] = mi;
            l[lm_offset + row_idx] = li;
        }
        block.sync();
    }
}

// Kernel de backward – versão escalar
__global__ void backward_kernel_scalar(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ S,
    const float* __restrict__ l,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int seq_len,
    const int embed_dim,
    const int total_col_blocks,
    const int total_row_blocks,
    const int block_size_col,
    const int block_size_row,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    
    int bx = blockIdx.x;  
    int by = blockIdx.y;  
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset  = (bx * gridDim.y * seq_len) + (by * seq_len);

    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* dO_tile = V_tile + block_size_col * embed_dim;
    float* S_tile = dO_tile + block_size_row * embed_dim;

    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d++) {
                int idx = qkv_offset + row_idx * embed_dim + d;
                Q_tile[tx * embed_dim + d] = Q[idx];
                dO_tile[tx * embed_dim + d] = dO[idx];
            }
        }

        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;

        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    K_tile[ty * embed_dim + d] = K[idx];
                    V_tile[ty * embed_dim + d] = V[idx];
                }
            }
            block.sync();

            float row_max = -INFINITY;
            if (tx < block_size_row && row_idx < seq_len) {
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[local_col * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + local_col] = sum;
                    row_max = fmaxf(row_max, sum);
                }
                float row_sum = 0.0f;
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float exp_val = expf(S_tile[tx * block_size_col + local_col] - row_max);
                    S_tile[tx * block_size_col + local_col] = exp_val;
                    row_sum += exp_val;
                }
                float sum_dS = 0.0f;
                for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                    float ds = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        ds += dO_tile[tx * embed_dim + d] * V_tile[local_col * embed_dim + d];
                    }
                    float p = S_tile[tx * block_size_col + local_col] / (li + 1e-10f);
                    ds *= scale;
                    sum_dS += ds * p;
                    S_tile[tx * block_size_col + local_col] = p * (ds - sum_dS);
                }
            }
            block.sync();

            if (tx < block_size_row && row_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    float acc = 0.0f;
                    for (int local_col = 0; local_col < block_size_col && (col_block * block_size_col + local_col) < seq_len; local_col++) {
                        float ds = S_tile[tx * block_size_col + local_col];
                        acc += ds * K_tile[local_col * embed_dim + d];
                    }
                    int idx = qkv_offset + row_idx * embed_dim + d;
                    atomicAdd(&dQ[idx], acc);
                }
            }

            if (ty < block_size_col && (col_block * block_size_col + ty) < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    float dk_acc = 0.0f;
                    float dv_acc = 0.0f;
                    for (int r = 0; r < block_size_row && (row_block * block_size_row + r) < seq_len; r++) {
                        float ds = S_tile[r * block_size_col + ty];
                        float q_val = Q_tile[r * embed_dim + d];
                        float do_val = dO_tile[r * embed_dim + d];
                        float p = S_tile[r * block_size_col + ty] / (l[lm_offset + (row_block * block_size_row + r)] + 1e-10f);
                        dk_acc += ds * q_val;
                        dv_acc += p * do_val;
                    }
                    int idx = qkv_offset + (col_block * block_size_col + ty) * embed_dim + d;
                    atomicAdd(&dK[idx], dk_acc);
                    atomicAdd(&dV[idx], dv_acc);
                }
            }
            block.sync();
        }
    }
}

/***************************************************
 * Estrutura para Gerenciamento de Memória no Device
 ***************************************************/
template <typename T>
struct DeviceArray {
    T* ptr;
    size_t size;
    
    DeviceArray(size_t s, bool zero_init = false) : size(s) {
        CUDA_CHECK(cudaMalloc(&ptr, size));
        if (zero_init)
            CUDA_CHECK(cudaMemset(ptr, 0, size));
    }
    
    ~DeviceArray() { cudaFree(ptr); }
    
    void curand_init(float min_val = 0.01f, float max_val = 0.1f) {
        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, time(0)));
        CURAND_CHECK(curandGenerateUniform(gen, ptr, size / sizeof(T)));
        
        int n = size / sizeof(T);
        float* temp;
        CUDA_CHECK(cudaMalloc(&temp, size));
        CUDA_CHECK(cudaMemcpy(temp, ptr, size, cudaMemcpyDeviceToDevice));
        
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scale_shift_kernel<<<blocks, threads>>>(temp, ptr, n, min_val, max_val - min_val);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaFree(temp));
        CURAND_CHECK(curandDestroyGenerator(gen));
    }
};

void print_matrix(float* matrix, int batch_size, int num_heads, int seq_len, int embed_dim, const char* name) {
    float* host_matrix = new float[batch_size * num_heads * seq_len * embed_dim];
    CUDA_CHECK(cudaMemcpy(host_matrix, matrix, batch_size * num_heads * seq_len * embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "\n" << name << ":\n";
    for (int b = 0; b < batch_size; b++) {
        std::cout << "Batch " << b << ":\n";
        for (int h = 0; h < num_heads; h++) {
            std::cout << "Head " << h << ":\n";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < embed_dim; j++) {
                    int idx = b * num_heads * seq_len * embed_dim + h * seq_len * embed_dim + i * embed_dim + j;
                    std::cout << host_matrix[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    delete[] host_matrix;
}

int main() {
    constexpr int batch_size = 1;
    constexpr int num_heads = 2;
    constexpr int seq_len = 4;
    // Teste: se embed_dim não for múltiplo de 4, as versões escalares serão usadas.
    const int embed_dim = 7;  
    constexpr int block_size_col = 4;
    constexpr int block_size_row = 4;
    
    const int total_col_blocks = (seq_len + block_size_col - 1) / block_size_col;
    const int total_row_blocks = (seq_len + block_size_row - 1) / block_size_row;
    const float scale = 1.0f / sqrtf(embed_dim);
    
    size_t matrix_size = batch_size * num_heads * seq_len * embed_dim * sizeof(float);
    size_t vector_size = batch_size * num_heads * seq_len * sizeof(float);
    
    DeviceArray<float> Q(matrix_size); Q.curand_init(0.01f, 0.1f);
    DeviceArray<float> K(matrix_size); K.curand_init(0.01f, 0.1f);
    DeviceArray<float> V(matrix_size); V.curand_init(0.01f, 0.1f);
    DeviceArray<float> O(matrix_size, true);
    DeviceArray<float> l(vector_size, true);
    DeviceArray<float> m(vector_size, true);
    // Inicializa m com -INFINITY usando o kernel
    {
        int n = vector_size / sizeof(float);
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        init_value_kernel<<<blocks, threads>>>(m.ptr, n, -INFINITY);
        CUDA_CHECK(cudaGetLastError());
    }
    
    DeviceArray<float> dO(matrix_size); dO.curand_init(0.01f, 0.1f);
    DeviceArray<float> dQ(matrix_size, true);
    DeviceArray<float> dK(matrix_size, true);
    DeviceArray<float> dV(matrix_size, true);
    DeviceArray<float> S(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    
    dim3 grid(batch_size, num_heads);
    dim3 block(block_size_row, block_size_col);
    
    // Tamanhos de memória compartilhada para os kernels
    size_t smem_size_forward_vector = (block_size_row * embed_dim +
                                2 * block_size_col * embed_dim +
                                block_size_row * block_size_col) * sizeof(float);
    size_t smem_size_backward_vector = (2 * block_size_row * embed_dim +
                                 2 * block_size_col * embed_dim +
                                 block_size_row * block_size_col) * sizeof(float);
    
    size_t smem_size_forward_scalar = smem_size_forward_vector; // mesmas regiões, carregamento escalar
    size_t smem_size_backward_scalar = smem_size_backward_vector;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    print_matrix(Q.ptr, batch_size, num_heads, seq_len, embed_dim, "Q");
    print_matrix(K.ptr, batch_size, num_heads, seq_len, embed_dim, "K");
    print_matrix(V.ptr, batch_size, num_heads, seq_len, embed_dim, "V");
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Seleção: se embed_dim for múltiplo de 4, usa versão vetorizada; senão, usa a versão escalar.
    if (embed_dim % 4 == 0) {
        forward_kernel_vectorized<<<grid, block, smem_size_forward_vector>>>(
            Q.ptr, K.ptr, V.ptr, O.ptr, l.ptr, m.ptr,
            seq_len, embed_dim, total_col_blocks, total_row_blocks,
            block_size_col, block_size_row, scale
        );
        CUDA_CHECK(cudaGetLastError());
    } else {
        forward_kernel_scalar<<<grid, block, smem_size_forward_scalar>>>(
            Q.ptr, K.ptr, V.ptr, O.ptr, l.ptr, m.ptr,
            seq_len, embed_dim, total_col_blocks, total_row_blocks,
            block_size_col, block_size_row, scale
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    print_matrix(O.ptr, batch_size, num_heads, seq_len, embed_dim, "Output (O)");
    print_matrix(dO.ptr, batch_size, num_heads, seq_len, embed_dim, "dO");
    
    if (embed_dim % 4 == 0) {
        backward_kernel_vectorized<<<grid, block, smem_size_backward_vector>>>(
            Q.ptr, K.ptr, V.ptr, dO.ptr, S.ptr, l.ptr,
            dQ.ptr, dK.ptr, dV.ptr,
            seq_len, embed_dim, total_col_blocks, total_row_blocks,
            block_size_col, block_size_row, scale
        );
        CUDA_CHECK(cudaGetLastError());
    } else {
        backward_kernel_scalar<<<grid, block, smem_size_backward_scalar>>>(
            Q.ptr, K.ptr, V.ptr, dO.ptr, S.ptr, l.ptr,
            dQ.ptr, dK.ptr, dV.ptr,
            seq_len, embed_dim, total_col_blocks, total_row_blocks,
            block_size_col, block_size_row, scale
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Tempo total de execução (forward + backward): " << ms << " ms\n";
    
    print_matrix(dQ.ptr, batch_size, num_heads, seq_len, embed_dim, "dQ");
    print_matrix(dK.ptr, batch_size, num_heads, seq_len, embed_dim, "dK");
    print_matrix(dV.ptr, batch_size, num_heads, seq_len, embed_dim, "dV");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
