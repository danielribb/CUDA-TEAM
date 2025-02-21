%%writefile teste.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void forward_kernel(
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
    int tx = threadIdx.x;
    
    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset = (bx * gridDim.y * seq_len) + (by * seq_len);
    
    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* S_tile = V_tile + block_size_col * embed_dim;
    
    const float eps = 1e-10f;
    
    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        // Load Q
        if (tx < block_size_row && (row_block * block_size_row + tx) < seq_len) {
            for (int d = 0; d < embed_dim; d += 4) {  // Changed to 4 for our small example
                int idx = qkv_offset + (row_block * block_size_row + tx) * embed_dim + d;
                if (d + 4 <= embed_dim) {
                    float4 q = *(float4*)&Q[idx];
                    *(float4*)&Q_tile[tx * embed_dim + d] = q;
                }
            }
        }
        
        float mi = m[lm_offset + row_block * block_size_row + tx];
        float li = l[lm_offset + row_block * block_size_row + tx];
        
        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            // Load K and V
            if (tx < block_size_col && (col_block * block_size_col + tx) < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    if (d + 4 <= embed_dim) {
                        int idx = qkv_offset + (col_block * block_size_col + tx) * embed_dim + d;
                        *(float4*)&K_tile[tx * embed_dim + d] = *(float4*)&K[idx];
                        *(float4*)&V_tile[tx * embed_dim + d] = *(float4*)&V[idx];
                    }
                }
            }
            block.sync();
            
            // Compute attention scores
            float row_max = -INFINITY;
            float row_sum = 0.0f;
            
            for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                float sum = 0.0f;
                for (int d = 0; d < embed_dim; d++) {
                    sum += Q_tile[tx * embed_dim + d] * K_tile[col_idx * embed_dim + d];
                }
                sum *= scale;
                S_tile[tx * block_size_col + col_idx] = sum;
                row_max = fmaxf(row_max, sum);
            }
            
            // Softmax
            for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                float val = expf(S_tile[tx * block_size_col + col_idx] - row_max);
                S_tile[tx * block_size_col + col_idx] = val;
                row_sum += val;
            }
            
            // Update running statistics
            float m_new = fmaxf(mi, row_max);
            float l_new = expf(mi - m_new) * li + expf(row_max - m_new) * row_sum;
            
            // Compute output
            if (tx < block_size_row && (row_block * block_size_row + tx) < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                        float p = S_tile[tx * block_size_col + col_idx];
                        float4 v = *(float4*)&V_tile[col_idx * embed_dim + d];
                        acc.x += p * v.x;
                        acc.y += p * v.y;
                        acc.z += p * v.z;
                        acc.w += p * v.w;
                    }
                    
                    int out_idx = qkv_offset + (row_block * block_size_row + tx) * embed_dim + d;
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
        
        if (tx < block_size_row && (row_block * block_size_row + tx) < seq_len) {
            m[lm_offset + row_block * block_size_row + tx] = mi;
            l[lm_offset + row_block * block_size_row + tx] = li;
        }
    }
}

__global__ void backward_kernel(
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
    
    int bx = blockIdx.x;  // batch
    int by = blockIdx.y;  // head
    int tx = threadIdx.x;
    
    int qkv_offset = (bx * gridDim.y * seq_len * embed_dim) + (by * seq_len * embed_dim);
    int lm_offset = (bx * gridDim.y * seq_len) + (by * seq_len);
    
    extern __shared__ float s[];
    float* Q_tile = s;
    float* K_tile = Q_tile + block_size_row * embed_dim;
    float* V_tile = K_tile + block_size_col * embed_dim;
    float* dO_tile = V_tile + block_size_col * embed_dim;
    float* S_tile = dO_tile + block_size_row * embed_dim;
    
    for (int row_block = 0; row_block < total_row_blocks; row_block++) {
        if (tx < block_size_row && (row_block * block_size_row + tx) < seq_len) {
            for (int d = 0; d < embed_dim; d += 4) {
                int idx = qkv_offset + (row_block * block_size_row + tx) * embed_dim + d;
                *(float4*)&Q_tile[tx * embed_dim + d] = *(float4*)&Q[idx];
                *(float4*)&dO_tile[tx * embed_dim + d] = *(float4*)&dO[idx];
            }
        }
        
        float li = l[lm_offset + row_block * block_size_row + tx];
        
        for (int col_block = 0; col_block < total_col_blocks; col_block++) {
            if (tx < block_size_col && (col_block * block_size_col + tx) < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    int idx = qkv_offset + (col_block * block_size_col + tx) * embed_dim + d;
                    *(float4*)&K_tile[tx * embed_dim + d] = *(float4*)&K[idx];
                    *(float4*)&V_tile[tx * embed_dim + d] = *(float4*)&V[idx];
                }
            }
            block.sync();
            
            float row_max = -INFINITY;
            for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                float sum = 0.0f;
                for (int d = 0; d < embed_dim; d++) {
                    sum += Q_tile[tx * embed_dim + d] * K_tile[col_idx * embed_dim + d];
                }
                sum *= scale;
                S_tile[tx * block_size_col + col_idx] = sum;
                row_max = fmaxf(row_max, sum);
            }
            
            float row_sum = 0.0f;
            for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                float exp_val = expf(S_tile[tx * block_size_col + col_idx] - row_max);
                S_tile[tx * block_size_col + col_idx] = exp_val;
                row_sum += exp_val;
            }
            
            float sum_dS = 0.0f;
            for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                float ds = 0.0f;
                for (int d = 0; d < embed_dim; d++) {
                    ds += dO_tile[tx * embed_dim + d] * V_tile[col_idx * embed_dim + d];
                }
                float p = S_tile[tx * block_size_col + col_idx] / (li + 1e-10f);
                ds *= scale;
                sum_dS += ds;
                S_tile[tx * block_size_col + col_idx] = p * (ds - sum_dS);
            }
            
            for (int d = 0; d < embed_dim; d += 4) {
                float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int col_idx = 0; col_idx < block_size_col && (col_block * block_size_col + col_idx) < seq_len; col_idx++) {
                    float ds = S_tile[tx * block_size_col + col_idx];
                    float4 key_val = *(float4*)&K_tile[col_idx * embed_dim + d];
                    acc.x += ds * key_val.x;
                    acc.y += ds * key_val.y;
                    acc.z += ds * key_val.z;
                    acc.w += ds * key_val.w;
                }
                int idx = qkv_offset + (row_block * block_size_row + tx) * embed_dim + d;
                if (tx < block_size_row && (row_block * block_size_row + tx) < seq_len) {
                    atomicAdd(&dQ[idx + 0], acc.x);
                    atomicAdd(&dQ[idx + 1], acc.y);
                    atomicAdd(&dQ[idx + 2], acc.z);
                    atomicAdd(&dQ[idx + 3], acc.w);
                }
            }
            
            if (tx < block_size_col && (col_block * block_size_col + tx) < seq_len) {
                for (int d = 0; d < embed_dim; d += 4) {
                    float4 dk_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float4 dv_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int r = 0; r < block_size_row && (row_block * block_size_row + r) < seq_len; r++) {
                        float ds = S_tile[r * block_size_col + tx];
                        float4 q = *(float4*)&Q_tile[r * embed_dim + d];
                        float4 do_val = *(float4*)&dO_tile[r * embed_dim + d];
                        float p = S_tile[r * block_size_col + tx] / (li + 1e-10f);
                        
                        dk_acc.x += ds * q.x;
                        dk_acc.y += ds * q.y;
                        dk_acc.z += ds * q.z;
                        dk_acc.w += ds * q.w;
                        
                        dv_acc.x += p * do_val.x;
                        dv_acc.y += p * do_val.y;
                        dv_acc.z += p * do_val.z;
                        dv_acc.w += p * do_val.w;
                    }
                    int idx = qkv_offset + (col_block * block_size_col + tx) * embed_dim + d;
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

template <typename T>
struct DeviceArray {
    T* ptr;
    size_t size;
    
    DeviceArray(size_t s, bool zero_init = false) : size(s) {
        cudaMalloc(&ptr, size);
        if (zero_init) cudaMemset(ptr, 0, size);
    }
    
    ~DeviceArray() { cudaFree(ptr); }
    
    void random_init() {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(0));
        curandGenerateUniform(gen, ptr, size / sizeof(T));
        curandDestroyGenerator(gen);
    }
};

void print_matrix(float* matrix, int batch_size, int num_heads, int seq_len, int embed_dim, const char* name) {
    float* host_matrix = new float[batch_size * num_heads * seq_len * embed_dim];
    cudaMemcpy(host_matrix, matrix, batch_size * num_heads * seq_len * embed_dim * sizeof(float), 
              cudaMemcpyDeviceToHost);
    
    std::cout << "\n" << name << ":\n";
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < embed_dim; j++) {
                    int idx = b * num_heads * seq_len * embed_dim + 
                             h * seq_len * embed_dim + 
                             i * embed_dim + j;
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
    constexpr int num_heads = 1;
    constexpr int seq_len = 3;
    constexpr int embed_dim = 4;
    constexpr int block_size_col = 3;
    constexpr int block_size_row = 3;
    
    const int total_col_blocks = (seq_len + block_size_col - 1) / block_size_col;
    const int total_row_blocks = (seq_len + block_size_row - 1) / block_size_row;
    const float scale = 1.0f / sqrtf(embed_dim);
    
    size_t matrix_size = batch_size * num_heads * seq_len * embed_dim * sizeof(float);
    size_t vector_size = batch_size * num_heads * seq_len * sizeof(float);
    
    DeviceArray<float> Q(matrix_size);
    DeviceArray<float> K(matrix_size);
    DeviceArray<float> V(matrix_size);
    DeviceArray<float> O(matrix_size, true);
    DeviceArray<float> l(vector_size, true);
    DeviceArray<float> m(vector_size);
    cudaMemset(m.ptr, 0xFF, vector_size);  // -inf
    
    DeviceArray<float> dO(matrix_size);
    DeviceArray<float> dQ(matrix_size, true);
    DeviceArray<float> dK(matrix_size, true);
    DeviceArray<float> dV(matrix_size, true);
    DeviceArray<float> S(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    
    float h_Q[12] = {1.0f, 2.0f, 4.0f, 1.0f,
                    4.0f, 1.0f, 2.0f, 1.0f,
                    1.0f, 3.0f, 1.0f, 4.0f};
                    
    float h_K[12] = {1.0f, 1.0f, 2.0f, 3.0f,
                    0.0f, 1.0f, 2.0f, 5.0f,
                    1.0f, 2.0f, 1.0f, 3.0f};
                    
    float h_V[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f};
                    
    float h_dO[12] = {0.1f, 0.2f, 0.3f, 0.4f,
                     0.5f, 0.6f, 0.7f, 0.8f,
                     0.9f, 1.0f, 1.1f, 1.2f};

    cudaMemcpy(Q.ptr, h_Q, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K.ptr, h_K, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V.ptr, h_V, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dO.ptr, h_dO, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 grid(batch_size, num_heads);
    dim3 block(block_size_row);
    size_t smem_size = (block_size_row * embed_dim + 
                       2 * block_size_col * embed_dim + 
                       block_size_row * block_size_col) * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    print_matrix(Q.ptr, batch_size, num_heads, seq_len, embed_dim, "Q");
    print_matrix(K.ptr, batch_size, num_heads, seq_len, embed_dim, "K");
    print_matrix(V.ptr, batch_size, num_heads, seq_len, embed_dim, "V");
    
    cudaEventRecord(start);
    forward_kernel<<<grid, block, smem_size>>>(
        Q.ptr, K.ptr, V.ptr, O.ptr, l.ptr, m.ptr,
        seq_len, embed_dim, total_col_blocks, total_row_blocks,
        block_size_col, block_size_row, scale
    );
    cudaDeviceSynchronize();
    
    backward_kernel<<<grid, block, smem_size>>>(
        Q.ptr, K.ptr, V.ptr, dO.ptr, S.ptr, l.ptr,
        dQ.ptr, dK.ptr, dV.ptr,
        seq_len, embed_dim, total_col_blocks, total_row_blocks,
        block_size_col, block_size_row, scale
    );
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total execution time: " << ms << " ms\n";
    
    print_matrix(O.ptr, batch_size, num_heads, seq_len, embed_dim, "Output (O)");
    print_matrix(dO.ptr, batch_size, num_heads, seq_len, embed_dim, "dO");
    print_matrix(dQ.ptr, batch_size, num_heads, seq_len, embed_dim, "dQ");
    print_matrix(dK.ptr, batch_size, num_heads, seq_len, embed_dim, "dK");
    print_matrix(dV.ptr, batch_size, num_heads, seq_len, embed_dim, "dV");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
