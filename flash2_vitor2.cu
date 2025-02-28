%%writefile flash.cu
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

inline void checkCuda(cudaError_t err, const char* msg = nullptr) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err);
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
        exit(1);
    }
}

inline void checkCurand(curandStatus_t err, const char* msg = nullptr) {
    if (err != CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND error";
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
        exit(1);
    }
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// Ferramentas de redução warp
// --------------------------------------------------------------------------
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Reduções por bloco usando warp
__inline__ __device__ float blockReduceMax(float val) {
    static __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceMax(val);
    if (!lane) smem[wid] = val;
    __syncthreads();
    float r = -INFINITY;
    if (threadIdx.x < (blockDim.x >> 5)) r = smem[lane];
    r = warpReduceMax(r);
    return r;
}
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (!lane) smem[wid] = val;
    __syncthreads();
    float r = 0.f;
    if (threadIdx.x < (blockDim.x >> 5)) r = smem[lane];
    r = warpReduceSum(r);
    return r;
}

// --------------------------------------------------------------------------
// Kernel PASSO 1: Para cada linha i de Q, computa dot(Q_i, K_j) em chunks
//   e acumula: maxScore(i), sumExp(i).
// --------------------------------------------------------------------------
__global__ void fa2_pass1(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    int N, int D,
    int chunkSize,
    float* __restrict__ L )
{
    // Cada bloco processa UMA linha i
    int i = blockIdx.x;
    if (i >= N) return;
    // Carrega Q(i,:) em registradores
    // (Opcional: se quisermos encaixar Q(i,:) em shared/regs)
    // Aqui simplificamos e apenas acessamos Q em loop.

    // Redução local do bloco
    float localMax = -INFINITY;
    float threadMax = -INFINITY;
    // Soma local (depois do exp, iremos fazer em 2 passos)
    // Mas pass1: Precisamos calcular max
    // Então, iremos varrer K em 'chunks'.
    for (int start = 0; start < N; start += chunkSize) {
        int chunkLen = (start + chunkSize <= N) ? chunkSize : (N - start);
        // Para cada Key j do chunk
        for (int j = 0; j < chunkLen; j++) {
            int keyIdx = start + j;
            // Dot
            float dot = 0.f;
            for (int c = threadIdx.x; c < D; c += blockDim.x) {
                dot += Q[i*D + c] * K[keyIdx*D + c];
            }
            // Redução local das threads
            dot = blockReduceSum(dot); // soma partial
            if (threadIdx.x == 0) {
                // Só a thread 0 do warp faz a comparação local?
                // Precisamos max, então:
                if (dot > threadMax) threadMax = dot;
            }
            __syncthreads();
        }
    }
    // Precisamos reduzir threadMax no bloco
    float finalMax = blockReduceMax(threadMax);
    if (threadIdx.x == 0) {
        L[i] = finalMax;      // salva maxScore(i)
        L[i + N] = 0.0f;      // zera sumExp(i) (vai ser usado no pass1.2)
    }
}

// --------------------------------------------------------------------------
// Kernel PASSO 1.2 (opcional): para computar sumExp(i) = sum_j exp(dotVal - max)
//   Podemos unificar pass1 e pass1.2, mas aqui deixamos separado para clareza.
// --------------------------------------------------------------------------
__global__ void fa2_pass1_sum(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    int N, int D,
    int chunkSize,
    float* __restrict__ L )
{
    int i = blockIdx.x;
    if (i >= N) return;
    float maxVal = L[i];   // recupera o valor do max
    float localSum = 0.f;

    for (int start = 0; start < N; start += chunkSize) {
        int chunkLen = (start + chunkSize <= N) ? chunkSize : (N - start);
        for (int j = 0; j < chunkLen; j++) {
            int keyIdx = start + j;
            float dot = 0.f;
            for (int c = threadIdx.x; c < D; c += blockDim.x) {
                dot += Q[i*D + c] * K[keyIdx*D + c];
            }
            dot = blockReduceSum(dot);
            if (threadIdx.x == 0) {
                float p = expf(dot - maxVal);
                localSum += p;
            }
            __syncthreads();
        }
    }
    float finalSum = blockReduceSum(localSum);
    if (threadIdx.x == 0) {
        // Salva sumExp(i) em L[i + N]
        L[i + N] = finalSum;
    }
}

// --------------------------------------------------------------------------
// Kernel PASSO 2: Recalcula dot(Q_i,K_j) - maxVal -> exp(...) / sum * V_j
// --------------------------------------------------------------------------
__global__ void fa2_pass2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    int N, int D,
    int chunkSize,
    float* __restrict__ O,
    float* __restrict__ L )
{
    int i = blockIdx.x;
    if (i >= N) return;

    float maxVal = L[i];
    float sumVal = L[i + N];
    float invSum = 1.f / sumVal;

    // Precisamos acumular O(i,:) = soma_j [ p_j * V_j ]
    // p_j = exp(dot(Q_i,K_j) - maxVal)/sumVal
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        O[i*D + c] = 0.f;
    }
    __syncthreads();

    for (int start = 0; start < N; start += chunkSize) {
        int chunkLen = (start + chunkSize <= N) ? chunkSize : (N - start);

        // Vamos acumular em registrador e depois somar atômico ou pass2
        for (int j = 0; j < chunkLen; j++) {
            int keyIdx = start + j;
            float dot = 0.f;
            for (int c = threadIdx.x; c < D; c += blockDim.x) {
                dot += Q[i*D + c] * K[keyIdx*D + c];
            }
            // Redução do dot no warp/bloco
            dot = blockReduceSum(dot);
            __syncthreads(); 
            // Só thread 0 tem dot "final"
            float p = 0.f;
            if (threadIdx.x == 0) {
                p = expf(dot - maxVal) * invSum;
            }
            // broadcast p
            p = __shfl_sync(0xffffffff, p, 0);

            // soma p*V_j
            for (int c = threadIdx.x; c < D; c += blockDim.x) {
                atomicAdd(O + i*D + c, p * V[keyIdx*D + c]);
            }
            __syncthreads();
        }
    }

    // Se quisermos gravar L[i] = logSumExp no final:
    if (threadIdx.x == 0) {
        L[i] = maxVal + logf(sumVal);
    }
}

// --------------------------------------------------------------------------
// Exemplo de “kernel Flash2-like”
void flash2_like(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    float*       d_L,
    int N, int D,
    int chunkSize,
    int blockSize)
{
    // Lançamos 1 bloco por linha
    dim3 grid(N), block(blockSize);

    // Passo 1: encontrar max
    fa2_pass1<<<grid, block>>>(d_Q, d_K, N, D, chunkSize, d_L);
    cudaDeviceSynchronize();

    // Passo 1.2: somar exp
    fa2_pass1_sum<<<grid, block>>>(d_Q, d_K, N, D, chunkSize, d_L);
    cudaDeviceSynchronize();

    // Passo 2: calcula O = ...
    fa2_pass2<<<grid, block>>>(d_Q, d_K, d_V, N, D, chunkSize, d_O, d_L);
    cudaDeviceSynchronize();
}

float dotProduct(const float* a, const float* b, size_t length) {
    float produto = 0.0f;
    for (size_t i = 0; i < length; ++i)
        produto += a[i] * b[i];
    return produto;
}

float* softmax(const float* scores, size_t n) {
    float* exp_scores = new float[n];
    float soma_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        exp_scores[i] = std::exp(scores[i]);
        soma_exp += exp_scores[i];
    }
    for (size_t i = 0; i < n; ++i)
        exp_scores[i] /= soma_exp;
    return exp_scores;
}

void attention(const float* queries, size_t num_queries, size_t dim,
               const float* keys, size_t num_keys,
               const float* values, size_t value_dim,
               float* output) {
    for (size_t i = 0; i < num_queries; ++i) {
        const float* query = queries + i * dim;
        float escala = std::sqrt(static_cast<float>(dim));

        float* scores = new float[num_keys];
        for (size_t j = 0; j < num_keys; ++j) {
            const float* key = keys + j * dim;
            scores[j] = dotProduct(query, key, dim) / escala;
        }

        float* pesos = softmax(scores, num_keys);

        float* out_query = output + i * value_dim;
        for (size_t k = 0; k < value_dim; ++k)
            out_query[k] = 0.0f;

        for (size_t j = 0; j < num_keys; ++j) {
            const float* value = values + j * value_dim;
            for (size_t k = 0; k < value_dim; ++k)
                out_query[k] += pesos[j] * value[k];
        }

        delete[] scores;
        delete[] pesos;
    }
}

int main() {
    static const int N  = 1024; // número de queries
    static const int D  = 1024; // dimensão
    static const int BC = 64;   // chunkSize
    size_t sz = (size_t)N * D;
    
    float *h_Q = new float[sz];
    float *h_K = new float[sz];
    float *h_V = new float[sz];
    float *h_O = new float[sz];
    float *h_L = new float[2*N]; // se for usar [max, sumExp] => 2*N
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    checkCuda(cudaMalloc(&d_Q, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_K, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_V, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_O, sz*sizeof(float)));
    // Precisamos de 2*N se quisermos guardar [max, sumExp], 
    // ou N se estamos guardando no arranjo [i, i+N].
    checkCuda(cudaMalloc(&d_L, 2*N*sizeof(float)));
    
    // Gera dados aleatórios com CURAND
    curandGenerator_t gen;
    checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    checkCurand(curandGenerateUniform(gen, d_Q, sz));
    checkCurand(curandGenerateUniform(gen, d_K, sz));
    checkCurand(curandGenerateUniform(gen, d_V, sz));
    
    checkCurand(curandDestroyGenerator(gen));

    // Lançamos os kernels
    // Exemplo: 1 bloco por linha (N blocos), 128 threads por bloco
    const int threadsPerBlock = 32;
    dim3 gridDim(N), blockDim(threadsPerBlock);

    // (Opcional) Criar eventos para medir tempo
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));

    // -------------------------
    // OPÇÃO A: Chamar cada kernel manualmente
    // -------------------------
    fa2_pass1<<<gridDim, blockDim>>>(d_Q, d_K, N, D, BC, d_L);
    checkCuda(cudaGetLastError());
    
    fa2_pass1_sum<<<gridDim, blockDim>>>(d_Q, d_K, N, D, BC, d_L);
    checkCuda(cudaGetLastError());
    
    fa2_pass2<<<gridDim, blockDim>>>(d_Q, d_K, d_V, N, D, BC, d_O, d_L);
    checkCuda(cudaGetLastError());

    

    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float ms = 0;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Tempo total GPU: " << ms/1000.0f << " s\n";

    

    // Copia resultado
    checkCuda(cudaMemcpy(h_O, d_O, sz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_Q, d_Q, sz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_K, d_K, sz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_V, d_V, sz*sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i+=200) {
        std::cout << "Resultado da atenção para a GPU query " << i + 1 << ": ";
        for (size_t j = 0; j < D; j+=200) {
            std::cout << h_O[i * D + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    float output[N * D];

    attention(h_Q, N, D, h_K, N, h_V, D, output);

    for (size_t i = 0; i < N; i+=200) {
        std::cout << "Resultado da atenção para a CPU query " << i + 1 << ": ";
        for (size_t j = 0; j < D; j+=200) {
            std::cout << output[i * D + j] << " ";
        }
        std::cout << std::endl;
    }


    // Libera
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_L;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);

    return 0;
}
