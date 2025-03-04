#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>



// Definições de constantes

#define WARP_SIZE 32

#define FULL_MASK 0xffffffff



// Kernel para calcular a atenção (forward pass)

template <int BLOCK_SIZE_Q, int BLOCK_SIZE_KV, int HEAD_DIM, bool IS_CAUSAL>

__global__ void attention_forward_kernel(

    const float* __restrict__ Q,

    const float* __restrict__ K,

    const float* __restrict__ V,

    float* __restrict__ O,

    float* __restrict__ M,

    const float softmax_scale,

    const int batch_size,

    const int num_heads,

    const int seq_len,

    const int stride_q_batch,

    const int stride_q_head,

    const int stride_q_seq,

    const int stride_q_dim,

    const int stride_k_batch,

    const int stride_k_head,

    const int stride_k_seq,

    const int stride_k_dim,

    const int stride_v_batch,

    const int stride_v_head,

    const int stride_v_seq,

    const int stride_v_dim,

    const int stride_o_batch,

    const int stride_o_head,

    const int stride_o_seq,

    const int stride_o_dim

) {

    // Identificar quais blocos processar

    const int block_index_q = blockIdx.x;

    const int index_batch_head = blockIdx.y;

    const int index_batch = index_batch_head / num_heads;

    const int index_head = index_batch_head % num_heads;

    

    // Calcular offsets

    const int q_offset = index_batch * stride_q_batch + index_head * stride_q_head;

    const int k_offset = index_batch * stride_k_batch + index_head * stride_k_head;

    const int v_offset = index_batch * stride_v_batch + index_head * stride_v_head;

    const int o_offset = index_batch * stride_o_batch + index_head * stride_o_head;

    

    // Cada thread processa um elemento em Q

    const int q_idx = threadIdx.x;

    const int seq_idx = block_index_q * BLOCK_SIZE_Q + q_idx;

    

    // Buffer compartilhado para Q, K, V

    __shared__ float q_block[BLOCK_SIZE_Q][HEAD_DIM];

    __shared__ float k_block[BLOCK_SIZE_KV][HEAD_DIM];

    __shared__ float v_block[BLOCK_SIZE_KV][HEAD_DIM];

    

    // Variáveis para rastrear máximos e somas

    float m_i = -INFINITY;

    float l_i = 0.0f;

    float acc[HEAD_DIM] = {0.0f};

    

    // Carregar bloco Q

    if (q_idx < BLOCK_SIZE_Q && seq_idx < seq_len) {

        #pragma unroll

        for (int d = 0; d < HEAD_DIM; d++) {

            q_block[q_idx][d] = Q[q_offset + seq_idx * stride_q_seq + d * stride_q_dim];

        }

    }

    

    __syncthreads();

    

    // Loop sobre blocos K, V

    for (int kv_block_start = 0; kv_block_start < seq_len; kv_block_start += BLOCK_SIZE_KV) {

        const int kv_limit = IS_CAUSAL ? min(kv_block_start + BLOCK_SIZE_KV, seq_idx + 1) : kv_block_start + BLOCK_SIZE_KV;

        

        // Pré-carregar blocos K, V

        if (q_idx < BLOCK_SIZE_KV && kv_block_start + q_idx < seq_len) {

            #pragma unroll

            for (int d = 0; d < HEAD_DIM; d++) {

                k_block[q_idx][d] = K[k_offset + (kv_block_start + q_idx) * stride_k_seq + d * stride_k_dim];

                v_block[q_idx][d] = V[v_offset + (kv_block_start + q_idx) * stride_v_seq + d * stride_v_dim];

            }

        }

        

        __syncthreads();

        

        // Processar elementos dentro do bloco atual

        if (q_idx < BLOCK_SIZE_Q && seq_idx < seq_len) {

            float qk_max = -INFINITY;

            float qk_tmp[BLOCK_SIZE_KV];

            

            // Calcular QK para este bloco

            for (int kv_idx = 0; kv_idx < min(BLOCK_SIZE_KV, seq_len - kv_block_start); kv_idx++) {

                const int abs_kv_idx = kv_block_start + kv_idx;

                

                // Verificar máscara causal

                if (!IS_CAUSAL || abs_kv_idx <= seq_idx) {

                    float qk = 0.0f;

                    

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        qk += q_block[q_idx][d] * k_block[kv_idx][d];

                    }

                    qk *= softmax_scale;

                    

                    qk_max = max(qk_max, qk);

                    qk_tmp[kv_idx] = qk;

                } else {

                    qk_tmp[kv_idx] = -INFINITY;

                }

            }

            

            // Ajustar com o máximo anterior

            const float m_new = max(m_i, qk_max);

            const float scale = expf(m_i - m_new);

            

            float l_new = l_i * scale;

            

            // Calcular softmax e atualizar acumuladores

            for (int kv_idx = 0; kv_idx < min(BLOCK_SIZE_KV, seq_len - kv_block_start); kv_idx++) {

                const int abs_kv_idx = kv_block_start + kv_idx;

                

                // Verificar máscara causal

                if (!IS_CAUSAL || abs_kv_idx <= seq_idx) {

                    const float p = expf(qk_tmp[kv_idx] - m_new);

                    l_new += p;

                    

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        acc[d] = acc[d] * scale + p * v_block[kv_idx][d];

                    }

                }

            }

            

            m_i = m_new;

            l_i = l_new;

        }

        

        __syncthreads();

    }

    

    // Normalizar e escrever resultados

    if (q_idx < BLOCK_SIZE_Q && seq_idx < seq_len) {

        // Armazenar valor máximo para backward pass

        M[index_batch * num_heads * seq_len + index_head * seq_len + seq_idx] = m_i + log(l_i);

        

        // Normalizar e escrever O

        for (int d = 0; d < HEAD_DIM; d++) {

            O[o_offset + seq_idx * stride_o_seq + d * stride_o_dim] = acc[d] / l_i;

        }

    }

}



// Kernel para pré-processar o backward pass

__global__ void attention_backward_preprocess_kernel(

    const float* __restrict__ O,

    const float* __restrict__ dO,

    float* __restrict__ D,

    const int seq_len,

    const int head_dim

) {

    const int block_index_q = blockIdx.x;

    const int index_batch_head = blockIdx.y;

    const int q_idx = threadIdx.x;

    const int seq_idx = block_index_q * blockDim.x + q_idx;

    

    if (seq_idx < seq_len) {

        float d_sum = 0.0f;

        

        for (int d = 0; d < head_dim; d++) {

            const int idx = index_batch_head * head_dim * seq_len + seq_idx * head_dim + d;

            d_sum += dO[idx] * O[idx];

        }

        

        D[index_batch_head * seq_len + seq_idx] = d_sum;

    }

}



// Kernel para calcular gradientes dQ

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, bool IS_CAUSAL>

__global__ void attention_backward_dq_kernel(

    const float* __restrict__ Q,

    const float* __restrict__ K,

    const float* __restrict__ V,

    const float* __restrict__ dO,

    const float* __restrict__ M,

    const float* __restrict__ D,

    float* __restrict__ dQ,

    const float softmax_scale,

    const int batch_size,

    const int num_heads,

    const int seq_len,

    const int stride_batch,

    const int stride_head,

    const int stride_seq,

    const int stride_dim

) {

    const int index_block_q = blockIdx.x;

    const int index_batch_head = blockIdx.z;

    const int index_batch = index_batch_head / num_heads;

    const int index_head = index_batch_head % num_heads;

    

    const int q_idx = threadIdx.x;

    const int d_idx = threadIdx.y;

    const int seq_idx = index_block_q * BLOCK_Q + q_idx;

    

    const int offset_batch_head = index_batch * stride_batch + index_head * stride_head;

    const int offset_batch_head_seq = index_batch_head * seq_len;

    

    __shared__ float q_block[BLOCK_Q][HEAD_DIM];

    __shared__ float do_block[BLOCK_Q][HEAD_DIM];

    __shared__ float m_block[BLOCK_Q];

    __shared__ float d_block[BLOCK_Q];

    

    float dq_acc = 0.0f;

    

    // Carregar dados Q, dO e valores auxiliares

    if (q_idx < BLOCK_Q && seq_idx < seq_len && d_idx < HEAD_DIM) {

        q_block[q_idx][d_idx] = Q[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim];

        do_block[q_idx][d_idx] = dO[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim];

        

        if (d_idx == 0) {

            m_block[q_idx] = M[offset_batch_head_seq + seq_idx];

            d_block[q_idx] = D[offset_batch_head_seq + seq_idx];

        }

    }

    

    __syncthreads();

    

    // Loop sobre blocos K, V

    for (int kv_block_start = 0; kv_block_start < seq_len; kv_block_start += BLOCK_KV) {

        __shared__ float k_block[BLOCK_KV][HEAD_DIM];

        __shared__ float v_block[BLOCK_KV][HEAD_DIM];

        

        // Carregar blocos K, V

        if (q_idx < BLOCK_KV && kv_block_start + q_idx < seq_len && d_idx < HEAD_DIM) {

            k_block[q_idx][d_idx] = K[offset_batch_head + (kv_block_start + q_idx) * stride_seq + d_idx * stride_dim];

            v_block[q_idx][d_idx] = V[offset_batch_head + (kv_block_start + q_idx) * stride_seq + d_idx * stride_dim];

        }

        

        __syncthreads();

        

        // Calcular contribuição para dQ

        if (q_idx < BLOCK_Q && seq_idx < seq_len && d_idx < HEAD_DIM) {

            for (int kv_idx = 0; kv_idx < min(BLOCK_KV, seq_len - kv_block_start); kv_idx++) {

                const int abs_kv_idx = kv_block_start + kv_idx;

                

                // Verificar máscara causal

                if (!IS_CAUSAL || seq_idx >= abs_kv_idx) {

                    float qk = 0.0f;

                    

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        qk += q_block[q_idx][d] * k_block[kv_idx][d];

                    }

                    qk *= softmax_scale;

                    

                    const float p = expf(qk - m_block[q_idx]);

                    

                    float dp = 0.0f;

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        dp += do_block[q_idx][d] * v_block[kv_idx][d];

                    }

                    

                    const float ds = p * (dp - d_block[q_idx]);

                    dq_acc += softmax_scale * ds * k_block[kv_idx][d_idx];

                }

            }

        }

        

        __syncthreads();

    }

    

    // Escrever gradientes dQ

    if (q_idx < BLOCK_Q && seq_idx < seq_len && d_idx < HEAD_DIM) {

        dQ[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim] = dq_acc;

    }

}



// Kernel para calcular gradientes dK e dV

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, bool IS_CAUSAL>

__global__ void attention_backward_dk_dv_kernel(

    const float* __restrict__ Q,

    const float* __restrict__ K,

    const float* __restrict__ V,

    const float* __restrict__ dO,

    const float* __restrict__ M,

    const float* __restrict__ D,

    float* __restrict__ dK,

    float* __restrict__ dV,

    const float softmax_scale,

    const int batch_size,

    const int num_heads,

    const int seq_len,

    const int stride_batch,

    const int stride_head,

    const int stride_seq,

    const int stride_dim

) {

    const int index_block_kv = blockIdx.x;

    const int index_batch_head = blockIdx.z;

    const int index_batch = index_batch_head / num_heads;

    const int index_head = index_batch_head % num_heads;

    

    const int kv_idx = threadIdx.x;

    const int d_idx = threadIdx.y;

    const int seq_idx = index_block_kv * BLOCK_KV + kv_idx;

    

    const int offset_batch_head = index_batch * stride_batch + index_head * stride_head;

    const int offset_batch_head_seq = index_batch_head * seq_len;

    

    float dk_acc = 0.0f;

    float dv_acc = 0.0f;

    

    __shared__ float k_block[BLOCK_KV][HEAD_DIM];

    __shared__ float v_block[BLOCK_KV][HEAD_DIM];

    

    // Carregar blocos K, V

    if (kv_idx < BLOCK_KV && seq_idx < seq_len && d_idx < HEAD_DIM) {

        k_block[kv_idx][d_idx] = K[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim];

        v_block[kv_idx][d_idx] = V[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim];

    }

    

    __syncthreads();

    

    // Loop sobre blocos Q

    for (int q_block_start = 0; q_block_start < seq_len; q_block_start += BLOCK_Q) {

        __shared__ float q_block[BLOCK_Q][HEAD_DIM];

        __shared__ float do_block[BLOCK_Q][HEAD_DIM];

        __shared__ float m_block[BLOCK_Q];

        __shared__ float d_block[BLOCK_Q];

        

        // Carregar bloco Q e dados relacionados

        if (kv_idx < BLOCK_Q && q_block_start + kv_idx < seq_len && d_idx < HEAD_DIM) {

            const int q_idx = q_block_start + kv_idx;

            q_block[kv_idx][d_idx] = Q[offset_batch_head + q_idx * stride_seq + d_idx * stride_dim];

            do_block[kv_idx][d_idx] = dO[offset_batch_head + q_idx * stride_seq + d_idx * stride_dim];

            

            if (d_idx == 0) {

                m_block[kv_idx] = M[offset_batch_head_seq + q_idx];

                d_block[kv_idx] = D[offset_batch_head_seq + q_idx];

            }

        }

        

        __syncthreads();

        

        // Calcular contribuições para dK e dV

        if (kv_idx < BLOCK_KV && seq_idx < seq_len && d_idx < HEAD_DIM) {

            for (int q_idx_offset = 0; q_idx_offset < min(BLOCK_Q, seq_len - q_block_start); q_idx_offset++) {

                const int q_idx = q_block_start + q_idx_offset;

                

                // Verificar máscara causal

                if (!IS_CAUSAL || q_idx >= seq_idx) {

                    float qk = 0.0f;

                    

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        qk += q_block[q_idx_offset][d] * k_block[kv_idx][d];

                    }

                    qk *= softmax_scale;

                    

                    const float p = expf(qk - m_block[q_idx_offset]);

                    

                    // Calcular dV

                    dv_acc += p * do_block[q_idx_offset][d_idx];

                    

                    // Calcular dP e dS para dK

                    float dp = 0.0f;

                    #pragma unroll

                    for (int d = 0; d < HEAD_DIM; d++) {

                        dp += v_block[kv_idx][d] * do_block[q_idx_offset][d];

                    }

                    

                    const float ds = p * (dp - d_block[q_idx_offset]);

                    dk_acc += softmax_scale * ds * q_block[q_idx_offset][d_idx];

                }

            }

        }

        

        __syncthreads();

    }

    

    // Escrever resultados

    if (kv_idx < BLOCK_KV && seq_idx < seq_len && d_idx < HEAD_DIM) {

        dK[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim] = dk_acc;

        dV[offset_batch_head + seq_idx * stride_seq + d_idx * stride_dim] = dv_acc;

    }

}



// Wrapper de C++ para chamar os kernels CUDA

std::vector<torch::Tensor> attention_forward(

    torch::Tensor Q,

    torch::Tensor K,

    torch::Tensor V,

    bool causal,

    float softmax_scale

) {

    // Obter dimensões

    auto batch_size = Q.size(0);

    auto num_heads = Q.size(1);

    auto seq_len = Q.size(2);

    auto head_dim = Q.size(3);

    

    // Criar tensores de saída

    auto O = torch::empty_like(Q);

    auto M = torch::empty({batch_size, num_heads, seq_len}, torch::dtype(torch::kFloat32).device(Q.device()));

    

    // Definir tamanhos de blocos

    const int BLOCK_SIZE_Q = 128;

    const int BLOCK_SIZE_KV = 64;

    

    // Configurar grade de threads

    dim3 grid(

        (seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q,

        batch_size * num_heads

    );

    dim3 block(BLOCK_SIZE_Q);

    

    // Chamar kernel apropriado com base em causal

    if (causal) {

        attention_forward_kernel<BLOCK_SIZE_Q, BLOCK_SIZE_KV, 64, true><<<grid, block>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            O.data_ptr<float>(),

            M.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3),

            K.stride(0),

            K.stride(1),

            K.stride(2),

            K.stride(3),

            V.stride(0),

            V.stride(1),

            V.stride(2),

            V.stride(3),

            O.stride(0),

            O.stride(1),

            O.stride(2),

            O.stride(3)

        );

    } else {

        attention_forward_kernel<BLOCK_SIZE_Q, BLOCK_SIZE_KV, 64, false><<<grid, block>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            O.data_ptr<float>(),

            M.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3),

            K.stride(0),

            K.stride(1),

            K.stride(2),

            K.stride(3),

            V.stride(0),

            V.stride(1),

            V.stride(2),

            V.stride(3),

            O.stride(0),

            O.stride(1),

            O.stride(2),

            O.stride(3)

        );

    }

    

    return {O, M};

}



std::vector<torch::Tensor> attention_backward(

    torch::Tensor Q,

    torch::Tensor K,

    torch::Tensor V,

    torch::Tensor O,

    torch::Tensor M,

    torch::Tensor dO,

    bool causal,

    float softmax_scale

) {

    // Obter dimensões

    auto batch_size = Q.size(0);

    auto num_heads = Q.size(1);

    auto seq_len = Q.size(2);

    auto head_dim = Q.size(3);

    

    // Criar tensores para gradientes

    auto dQ = torch::zeros_like(Q);

    auto dK = torch::zeros_like(K);

    auto dV = torch::zeros_like(V);

    auto D = torch::empty({batch_size, num_heads, seq_len}, torch::dtype(torch::kFloat32).device(Q.device()));

    

    // Constantes de tamanho de bloco

    const int BLOCK_MACRO = 128;

    const int BLOCK_MICRO = 32;

    const int NUM_THREADS_PER_BLOCK = 256;

    

    // Chamar kernel de pré-processamento

    dim3 preprocess_grid((seq_len + BLOCK_MACRO - 1) / BLOCK_MACRO, batch_size * num_heads);

    dim3 preprocess_block(BLOCK_MACRO);

    

    attention_backward_preprocess_kernel<<<preprocess_grid, preprocess_block>>>(

        O.data_ptr<float>(),

        dO.data_ptr<float>(),

        D.data_ptr<float>(),

        seq_len,

        head_dim

    );

    

    // Configurar blocos para dK, dV

    dim3 grid_dk_dv((seq_len + BLOCK_MACRO - 1) / BLOCK_MACRO, 1, batch_size * num_heads);

    dim3 block_dk_dv(BLOCK_MICRO, head_dim > BLOCK_MICRO ? BLOCK_MICRO : head_dim);

    

    // Chamar kernel dK, dV

    if (causal) {

        attention_backward_dk_dv_kernel<BLOCK_MICRO, BLOCK_MACRO, 64, true><<<grid_dk_dv, block_dk_dv>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            dO.data_ptr<float>(),

            M.data_ptr<float>(),

            D.data_ptr<float>(),

            dK.data_ptr<float>(),

            dV.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3)

        );

    } else {

        attention_backward_dk_dv_kernel<BLOCK_MICRO, BLOCK_MACRO, 64, false><<<grid_dk_dv, block_dk_dv>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            dO.data_ptr<float>(),

            M.data_ptr<float>(),

            D.data_ptr<float>(),

            dK.data_ptr<float>(),

            dV.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3)

        );

    }

    

    // Configurar blocos para dQ

    dim3 grid_dq((seq_len + BLOCK_MACRO - 1) / BLOCK_MACRO, 1, batch_size * num_heads);

    dim3 block_dq(BLOCK_MICRO, head_dim > BLOCK_MICRO ? BLOCK_MICRO : head_dim);

    

    // Chamar kernel dQ

    if (causal) {

        attention_backward_dq_kernel<BLOCK_MACRO, BLOCK_MICRO, 64, true><<<grid_dq, block_dq>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            dO.data_ptr<float>(),

            M.data_ptr<float>(),

            D.data_ptr<float>(),

            dQ.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3)

        );

    } else {

        attention_backward_dq_kernel<BLOCK_MACRO, BLOCK_MICRO, 64, false><<<grid_dq, block_dq>>>(

            Q.data_ptr<float>(),

            K.data_ptr<float>(),

            V.data_ptr<float>(),

            dO.data_ptr<float>(),

            M.data_ptr<float>(),

            D.data_ptr<float>(),

            dQ.data_ptr<float>(),

            softmax_scale,

            batch_size,

            num_heads,

            seq_len,

            Q.stride(0),

            Q.stride(1),

            Q.stride(2),

            Q.stride(3)

        );

    }

    

    return {dQ, dK, dV};

}

