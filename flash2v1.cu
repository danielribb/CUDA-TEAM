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

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CURAND_CHECK(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "CURANDassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

__global__ void scale_shift_kernel(float* input, float* output, int n, float min_val, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = min_val + input[idx] * scale;
    }
}

// Forward FlashAttention-2
__global__ void flash_attention_2_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l,
    float* __restrict__ m,
    const int seq_len,
    const int embed_dim,
    const int block_size_row,
    const int block_size_col,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    int bx = blockIdx.x;  // batch
    int by = blockIdx.y;  // head
    int tx = threadIdx.x; // índice da linha
    int ty = threadIdx.y; // índice da coluna

    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset = (bx * gridDim.y * seq_len) + (by * seq_len);

    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* S_tile = V_tile + block_size_col * embed_dim;

    const float eps = 1e-10f;
    int total_row_blocks = (seq_len + block_size_row - 1) / block_size_row;
    int total_col_blocks = (seq_len + block_size_col - 1) / block_size_col;

    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d++) {
                Q_tile[tx * embed_dim + d] = Q[qkv_offset + row_idx * embed_dim + d];
            }
        }

        float mi = (row_idx < seq_len) ? m[lm_offset + row_idx] : -INFINITY;
        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;

        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    K_tile[ty * embed_dim + d] = K[qkv_offset + col_idx * embed_dim + d];
                    V_tile[ty * embed_dim + d] = V[qkv_offset + col_idx * embed_dim + d];
                }
            }
            block.sync();

            if (tx < block_size_row && row_idx < seq_len) {
                float row_max = -INFINITY;
                float row_sum = 0.0f;
                // Calcular S = Q · K^T em blocos
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[j * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + j] = sum;
                    row_max = fmaxf(row_max, sum);
                }

                // Softmax incremental
                float m_new = fmaxf(mi, row_max);
                float l_new = expf(mi - m_new) * li + expf(row_max - m_new);
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    S_tile[tx * block_size_col + j] = expf(S_tile[tx * block_size_col + j] - m_new);
                    row_sum += S_tile[tx * block_size_col + j];
                }
                l_new *= row_sum;

                // Atualizar O incrementalmente
                for (int d = 0; d < embed_dim; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                        acc += S_tile[tx * block_size_col + j] * V_tile[j * embed_dim + d];
                    }
                    int out_idx = qkv_offset + row_idx * embed_dim + d;
                    float o_prev = O[out_idx];
                    O[out_idx] = (li * expf(mi - m_new) * o_prev + expf(row_max - m_new) * acc) / (l_new + eps);
                }

                mi = m_new;
                li = l_new;
            }
            block.sync();

            if (tx < block_size_row && row_idx < seq_len) {
                m[lm_offset + row_idx] = mi;
                l[lm_offset + row_idx] = li;
            }
        }
    }
}

// Backward FlashAttention-2
__global__ void flash_attention_2_backward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ l,
    const float* __restrict__ m,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int seq_len,
    const int embed_dim,
    const int block_size_row,
    const int block_size_col,
    const float scale)
{
    cg::thread_block block = cg::this_thread_block();
    int bx = blockIdx.x;  // batch
    int by = blockIdx.y;  // head
    int tx = threadIdx.x; // índice da linha
    int ty = threadIdx.y; // índice da coluna

    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset = (bx * gridDim.y * seq_len) + (by * seq_len);

    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* dO_tile = V_tile + block_size_col * embed_dim;
    float* S_tile = dO_tile + block_size_row * embed_dim;

    const float eps = 1e-10f;
    int total_row_blocks = (seq_len + block_size_row - 1) / block_size_row;
    int total_col_blocks = (seq_len + block_size_col - 1) / block_size_col;

    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        int row_idx = row_block * block_size_row + tx;
        if (tx < block_size_row && row_idx < seq_len) {
            for (int d = 0; d < embed_dim; d++) {
                Q_tile[tx * embed_dim + d] = Q[qkv_offset + row_idx * embed_dim + d];
                dO_tile[tx * embed_dim + d] = dO[qkv_offset + row_idx * embed_dim + d];
            }
        }

        float mi = (row_idx < seq_len) ? m[lm_offset + row_idx] : -INFINITY;
        float li = (row_idx < seq_len) ? l[lm_offset + row_idx] : 0.0f;

        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            int col_idx = col_block * block_size_col + ty;
            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    K_tile[ty * embed_dim + d] = K[qkv_offset + col_idx * embed_dim + d];
                    V_tile[ty * embed_dim + d] = V[qkv_offset + col_idx * embed_dim + d];
                }
            }
            block.sync();

            if (tx < block_size_row && row_idx < seq_len) {
                float row_max = -INFINITY;
                // Recomputar S
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        sum += Q_tile[tx * embed_dim + d] * K_tile[j * embed_dim + d];
                    }
                    sum *= scale;
                    S_tile[tx * block_size_col + j] = sum;
                    row_max = fmaxf(row_max, sum);
                }

                float row_sum = 0.0f;
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    float val = expf(S_tile[tx * block_size_col + j] - row_max);
                    S_tile[tx * block_size_col + j] = val;
                    row_sum += val;
                }
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    S_tile[tx * block_size_col + j] /= (row_sum + eps);
                }

                // Calcular dS
                float sum_dS = 0.0f;
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    float ds = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        ds += dO_tile[tx * embed_dim + d] * V_tile[j * embed_dim + d];
                    }
                    sum_dS += ds * S_tile[tx * block_size_col + j];
                }
                for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                    float ds = 0.0f;
                    for (int d = 0; d < embed_dim; d++) {
                        ds += dO_tile[tx * embed_dim + d] * V_tile[j * embed_dim + d];
                    }
                    S_tile[tx * block_size_col + j] = scale * S_tile[tx * block_size_col + j] * (ds - sum_dS);
                }

                // Calcular dQ
                for (int d = 0; d < embed_dim; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < block_size_col && (col_block * block_size_col + j) < seq_len; j++) {
                        acc += S_tile[tx * block_size_col + j] * K_tile[j * embed_dim + d];
                    }
                    int idx = qkv_offset + row_idx * embed_dim + d;
                    atomicAdd(&dQ[idx], acc);
                }
            }
            block.sync();

            if (ty < block_size_col && col_idx < seq_len) {
                for (int d = 0; d < embed_dim; d++) {
                    float dk_acc = 0.0f;
                    float dv_acc = 0.0f;
                    for (int r = 0; r < block_size_row && (row_block * block_size_row + r) < seq_len; r++) {
                        float ds = S_tile[r * block_size_col + ty];
                        dk_acc += ds * Q_tile[r * embed_dim + d];
                        dv_acc += S_tile[r * block_size_col + ty] * dO_tile[r * embed_dim + d];
                    }
                    int idx = qkv_offset + col_idx * embed_dim + d;
                    atomicAdd(&dK[idx], dk_acc);
                    atomicAdd(&dV[idx], dv_acc);
                }
            }
            block.sync();
        }
    }
}

template <typename T>
struct DeviceArray {
    T* ptr;
    size_t size;
    
    DeviceArray(size_t s, bool zero_init = false) : size(s) {
        CUDA_CHECK(cudaMalloc(&ptr, size));
        if (zero_init) CUDA_CHECK(cudaMemset(ptr, 0, size));
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
    constexpr int batch_size = 2;
    constexpr int num_heads = 8;
    constexpr int seq_len = 512;
    const int embed_dim = 1024;  
    constexpr int block_size_col = 64;
    constexpr int block_size_row = 64;
    
    const float scale = 1.0f / sqrtf(embed_dim);
    
    size_t matrix_size = batch_size * num_heads * seq_len * embed_dim * sizeof(float);
    size_t vector_size = batch_size * num_heads * seq_len * sizeof(float);
    
    DeviceArray<float> Q(matrix_size); Q.curand_init(0.01f, 0.1f);
    DeviceArray<float> K(matrix_size); K.curand_init(0.01f, 0.1f);
    DeviceArray<float> V(matrix_size); V.curand_init(0.01f, 0.1f);
    DeviceArray<float> O(matrix_size, true);
    DeviceArray<float> l(vector_size, true);
    DeviceArray<float> m(vector_size);
    CUDA_CHECK(cudaMemset(m.ptr, *reinterpret_cast<int*>(&(-INFINITY)), vector_size));
    
    DeviceArray<float> dO(matrix_size); dO.curand_init(0.01f, 0.1f);
    DeviceArray<float> dQ(matrix_size, true);
    DeviceArray<float> dK(matrix_size, true);
    DeviceArray<float> dV(matrix_size, true);
    
    dim3 grid(batch_size, num_heads);
    dim3 block(block_size_row, block_size_col);
    size_t smem_size = (block_size_row * embed_dim + 
                        2 * block_size_col * embed_dim + 
                        block_size_row * block_size_col) * sizeof(float);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    print_matrix(Q.ptr, batch_size, num_heads, seq_len, embed_dim, "Q");
    print_matrix(K.ptr, batch_size, num_heads, seq_len, embed_dim, "K");
    print_matrix(V.ptr, batch_size, num_heads, seq_len, embed_dim, "V");
    
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_2_forward<<<grid, block, smem_size>>>(
        Q.ptr, K.ptr, V.ptr, O.ptr, l.ptr, m.ptr,
        seq_len, embed_dim, block_size_row, block_size_col, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    print_matrix(O.ptr, batch_size, num_heads, seq_len, embed_dim, "Output (O)");
    print_matrix(dO.ptr, batch_size, num_heads, seq_len, embed_dim, "dO");
    
    flash_attention_2_backward<<<grid, block, smem_size>>>(
        Q.ptr, K.ptr, V.ptr, dO.ptr, l.ptr, m.ptr,
        dQ.ptr, dK.ptr, dV.ptr,
        seq_len, embed_dim, block_size_row, block_size_col, scale
    );
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
