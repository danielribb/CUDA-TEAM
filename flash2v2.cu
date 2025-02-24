#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define EPSILON 1e-6f 
#define BLOCK_SIZE_Q 64  
#define BLOCK_SIZE_KV 32   
#define THREADS_PER_BLOCK 128  
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}
#define CHECK_CURAND(call) { \
    curandStatus_t err = call; \
    if (err != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND error in %s:%d - %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

__global__ void scale_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f - 1.0f;
    }
}

__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* M,
    int batch_size, int n_heads, int seq_len,
    int head_dim, int block_size_q, int block_size_kv,
    float softmax_scale, bool causal) {
    int b = blockIdx.y / n_heads;
    int h = blockIdx.y % n_heads;
    int block_idx_q = blockIdx.x;
    int tid = threadIdx.x;
    int q_start = block_idx_q * block_size_q;
    if (q_start >= seq_len) return;
    
    __shared__ float Q_block[BLOCK_SIZE_Q][THREADS_PER_BLOCK];
    __shared__ float S[BLOCK_SIZE_Q][BLOCK_SIZE_KV];
    __shared__ float P[BLOCK_SIZE_Q][BLOCK_SIZE_KV];
    
    if (tid < head_dim && q_start + blockIdx.z < seq_len) {
        int q_idx = b * n_heads * seq_len * head_dim + 
                   h * seq_len * head_dim + 
                   (q_start + blockIdx.z) * head_dim + tid;
        Q_block[blockIdx.z][tid] = Q[q_idx];
    }
    __syncthreads();
    
    float l_i = 0.0f;
    float m_i = -INFINITY;
    float O_block[THREADS_PER_BLOCK] = {0.0f};
    
    for (int kv_start = 0; kv_start < seq_len; kv_start += block_size_kv) {
        int kv_end = min(kv_start + block_size_kv, seq_len);
        if (tid < block_size_kv && q_start + blockIdx.z < seq_len && kv_start + tid < seq_len) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int k_idx = b * n_heads * seq_len * head_dim + 
                           h * seq_len * head_dim + 
                           (kv_start + tid) * head_dim + d;
                score += Q_block[blockIdx.z][d] * K[k_idx];
            }
            score *= softmax_scale;
            if (causal && (q_start + blockIdx.z) < (kv_start + tid)) {
                score = -1.0e6;
            }
            S[blockIdx.z][tid] = score;
            m_i = max(m_i, score);
        }
        __syncthreads();
        
        if (tid < block_size_kv && q_start + blockIdx.z < seq_len) {
            float p = expf(S[blockIdx.z][tid] - m_i);
            P[blockIdx.z][tid] = p;
            l_i += p;
        }
        __syncthreads();
        
        if (tid < head_dim && q_start + blockIdx.z < seq_len) {
            for (int k = 0; k < block_size_kv && (kv_start + k) < seq_len; k++) {
                int v_idx = b * n_heads * seq_len * head_dim + 
                           h * seq_len * head_dim + 
                           (kv_start + k) * head_dim + tid;
                O_block[tid] += P[blockIdx.z][k] * V[v_idx];
            }
        }
        __syncthreads();
    }
    
    if (tid < head_dim && q_start + blockIdx.z < seq_len) {
        int o_idx = b * n_heads * seq_len * head_dim + 
                   h * seq_len * head_dim + 
                   (q_start + blockIdx.z) * head_dim + tid;
        O[o_idx] = O_block[tid] / l_i;
        if (tid == 0) {
            M[b * n_heads * seq_len + h * seq_len + (q_start + blockIdx.z)] = m_i + logf(l_i);
        }
    }
}

__global__ void flash_attention_backward_preprocess(
    const float* O, const float* dO, float* D,
    int batch_size, int n_heads, int seq_len, int head_dim) {
    int b = blockIdx.y / n_heads;
    int h = blockIdx.y % n_heads;
    int q_start = blockIdx.x * BLOCK_SIZE_Q;
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    if (tid < head_dim && q_start + blockIdx.z < seq_len) {
        int idx = b * n_heads * seq_len * head_dim + 
                 h * seq_len * head_dim + 
                 (q_start + blockIdx.z) * head_dim + tid;
        float o = O[idx];
        float do_val = dO[idx];
        atomicAdd(&D[b * n_heads * seq_len + h * seq_len + (q_start + blockIdx.z)], o * do_val);
    }
}

__global__ void flash_attention_backward_dq(
    const float* Q, const float* K, const float* V,
    const float* dO, float* dQ, const float* M, const float* D,
    int batch_size, int n_heads, int seq_len, 
    int head_dim, float softmax_scale, bool causal) {
    int b = blockIdx.y / n_heads;
    int h = blockIdx.y % n_heads;
    int q_start = blockIdx.x * BLOCK_SIZE_Q;
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    __shared__ float Q_block[BLOCK_SIZE_Q][THREADS_PER_BLOCK];
    __shared__ float dO_block[BLOCK_SIZE_Q][THREADS_PER_BLOCK];
    
    if (tid < head_dim && q_start + blockIdx.z < seq_len) {
        int idx = b * n_heads * seq_len * head_dim + 
                 h * seq_len * head_dim + 
                 (q_start + blockIdx.z) * head_dim + tid;
        Q_block[blockIdx.z][tid] = Q[idx];
        dO_block[blockIdx.z][tid] = dO[idx];
    }
    __syncthreads();
    
    float dQ_block[THREADS_PER_BLOCK] = {0.0f};
    float m = M[b * n_heads * seq_len + h * seq_len + (q_start + blockIdx.z)];
    float delta = D[b * n_heads * seq_len + h * seq_len + (q_start + blockIdx.z)];  
    
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_SIZE_KV) {
        if (tid < BLOCK_SIZE_KV && q_start + blockIdx.z < seq_len && kv_start + tid < seq_len) {
            float qk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int k_idx = b * n_heads * seq_len * head_dim + 
                           h * seq_len * head_dim + 
                           (kv_start + tid) * head_dim + d;
                qk += Q_block[blockIdx.z][d] * K[k_idx];
            }
            qk *= softmax_scale;
            float p = expf(qk - m);
            if (causal && (q_start + blockIdx.z) < (kv_start + tid)) p = 0.0f;
            float dp = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int v_idx = b * n_heads * seq_len * head_dim + 
                           h * seq_len * head_dim + 
                           (kv_start + tid) * head_dim + d;
                dp += dO_block[blockIdx.z][d] * V[v_idx];
            }
            float ds = p * (dp - delta); 
            for (int d = 0; d < head_dim; d++) {
                int k_idx = b * n_heads * seq_len * head_dim + 
                           h * seq_len * head_dim + 
                           (kv_start + tid) * head_dim + d;
                dQ_block[d] += softmax_scale * ds * K[k_idx];
            }
        }
        __syncthreads();
    }
    
    if (tid < head_dim && q_start + blockIdx.z < seq_len) {
        int dq_idx = b * n_heads * seq_len * head_dim + 
                    h * seq_len * head_dim + 
                    (q_start + blockIdx.z) * head_dim + tid;
        dQ[dq_idx] = dQ_block[tid];
    }
}

void generate_random_matrix(float* d_matrix, int size, curandGenerator_t gen) {
    CHECK_CURAND(curandGenerateUniform(gen, d_matrix, size));
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_matrix, size);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float h_check[4];
    CHECK_CUDA(cudaMemcpy(h_check, d_matrix, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Random matrix check: %.4f %.4f %.4f %.4f\n", h_check[0], h_check[1], h_check[2], h_check[3]);
}

void print_matrix(const char* name, float* matrix, int batch_size, int n_heads, 
                 int seq_len, int dim) {
    float* h_matrix = (float*)malloc(batch_size * n_heads * seq_len * dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_matrix, matrix, 
                         batch_size * n_heads * seq_len * dim * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    printf("\n%s:\n", name);
    for (int b = 0; b < min(1, batch_size); b++) {
        for (int h = 0; h < min(1, n_heads); h++) {
            for (int i = 0; i < min(4, seq_len); i++) {
                for (int j = 0; j < min(4, dim); j++) {
                    int idx = b * n_heads * seq_len * dim + h * seq_len * dim + i * dim + j;
                    printf("%.4f ", h_matrix[idx]);
                }
                printf("... ");
            }
            printf("\n");
        }
    }
    free(h_matrix);
}


void attention_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& O,
    int batch_size, int n_heads, int seq_len, int head_dim, bool causal) {

    float softmax_scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                std::vector<float> scores(seq_len, 0.0f);

                for (int j = 0; j < seq_len; ++j) {
                    if (causal && j > i) {
                        scores[j] = -1e6f;  
                        continue;
                    }
                    float dot_product = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        int q_idx = ((b * n_heads + h) * seq_len + i) * head_dim + d;
                        int k_idx = ((b * n_heads + h) * seq_len + j) * head_dim + d;
                        dot_product += Q[q_idx] * K[k_idx];
                    }
                    scores[j] = dot_product * softmax_scale;
                }

                float max_score = -INFINITY;
                for (int j = 0; j < seq_len; ++j)
                    max_score = fmaxf(max_score, scores[j]);

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < seq_len; ++j)
                    scores[j] /= sum_exp;

                for (int d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        int v_idx = ((b * n_heads + h) * seq_len + j) * head_dim + d;
                        sum += scores[j] * V[v_idx];
                    }
                    int o_idx = ((b * n_heads + h) * seq_len + i) * head_dim + d;
                    O[o_idx] = sum;
                }
            }
        }
    }
}

bool compare_outputs(const std::vector<float>& A, const std::vector<float>& B, float tolerance = EPSILON) {
    for (size_t i = 0; i < A.size(); ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Diff at index %zu: CPU %.6f, CUDA %.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void copy_from_gpu(float* d_data, std::vector<float>& h_data, size_t size) {
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void verify_attention_results(float* d_Q, float* d_K, float* d_V, float* d_O, 
                              int batch_size, int n_heads, int seq_len, int head_dim, bool causal) {

    size_t total_size = batch_size * n_heads * seq_len * head_dim;
    std::vector<float> Q(total_size), K(total_size), V(total_size), O_CUDA(total_size), O_CPU(total_size);

    copy_from_gpu(d_Q, Q, total_size);
    copy_from_gpu(d_K, K, total_size);
    copy_from_gpu(d_V, V, total_size);
    copy_from_gpu(d_O, O_CUDA, total_size);

    printf("Running attention on CPU...\n");
    attention_cpu(Q, K, V, O_CPU, batch_size, n_heads, seq_len, head_dim, causal);

    if (compare_outputs(O_CPU, O_CUDA)) {
        printf("Atenção CUDA e CPU são equivalentes!\n");
    } else {
        printf("Diferenças detectadas entre CUDA e CPU!\n");
    }
}
 
int main() {
    int batch_size = 8;
    int n_heads = 16;
    int seq_len = 1024;
    int head_dim = 64;
    bool causal = true;
    float softmax_scale = 1.0f / sqrtf((float)head_dim);

    size_t qkv_size = batch_size * n_heads * seq_len * head_dim * sizeof(float);
    size_t m_size = batch_size * n_heads * seq_len * sizeof(float);
    
    float *d_Q, *d_K, *d_V, *d_O, *d_M, *d_dO, *d_D, *d_dQ;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_M, m_size));
    CHECK_CUDA(cudaMalloc(&d_dO, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_D, m_size));
    CHECK_CUDA(cudaMalloc(&d_dQ, qkv_size));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    printf("Generating random matrices...\n");
    generate_random_matrix(d_Q, batch_size * n_heads * seq_len * head_dim, gen);
    generate_random_matrix(d_K, batch_size * n_heads * seq_len * head_dim, gen);
    generate_random_matrix(d_V, batch_size * n_heads * seq_len * head_dim, gen);
    generate_random_matrix(d_dO, batch_size * n_heads * seq_len * head_dim, gen);

    CHECK_CUDA(cudaMemset(d_O, 0, qkv_size));

    CHECK_CUDA(cudaMemset(d_M, 0, m_size));
    CHECK_CUDA(cudaMemset(d_D, 0, m_size));
    CHECK_CUDA(cudaMemset(d_dQ, 0, qkv_size));

    dim3 grid_forward((seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q, batch_size * n_heads, BLOCK_SIZE_Q);
    printf("Launching forward kernel...\n");
    flash_attention_forward<<<grid_forward, THREADS_PER_BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_M, batch_size, n_heads, seq_len, 
        head_dim, BLOCK_SIZE_Q, BLOCK_SIZE_KV, softmax_scale, causal);
    CHECK_CUDA(cudaDeviceSynchronize());
    verify_attention_results(d_Q, d_K, d_V, d_O, batch_size, n_heads, seq_len, head_dim, causal);
    dim3 grid_preprocess((seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q, batch_size * n_heads, BLOCK_SIZE_Q);
    printf("Launching backward preprocess kernel...\n");
    flash_attention_backward_preprocess<<<grid_preprocess, THREADS_PER_BLOCK>>>(
        d_O, d_dO, d_D, batch_size, n_heads, seq_len, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 grid_dq((seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q, batch_size * n_heads, BLOCK_SIZE_Q);
    printf("Launching backward dQ kernel...\n");
    flash_attention_backward_dq<<<grid_dq, THREADS_PER_BLOCK>>>(
        d_Q, d_K, d_V, d_dO, d_dQ, d_M, d_D, batch_size, n_heads, seq_len,
        head_dim, softmax_scale, causal);
    CHECK_CUDA(cudaDeviceSynchronize());

   // print_matrix("Q", d_Q, batch_size, n_heads, seq_len, head_dim);
   //print_matrix("K", d_K, batch_size, n_heads, seq_len, head_dim);
    //print_matrix("V", d_V, batch_size, n_heads, seq_len, head_dim);
    print_matrix("O", d_O, batch_size, n_heads, seq_len, head_dim);
   // print_matrix("dQ", d_dQ, batch_size, n_heads, seq_len, head_dim);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_dO));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_dQ));
    CHECK_CURAND(curandDestroyGenerator(gen));

    printf("\nDone!\n");
    return 0;
}
