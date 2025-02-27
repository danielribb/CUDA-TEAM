%%writefile flash.cu
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <chrono>

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

__global__ void optimized_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, int D, int BC)
{
    // Cada bloco processa uma query
    int qIdx = blockIdx.x;
    int tid  = threadIdx.x;
    int blockSize = blockDim.x;

    // Cada thread é responsável por uma parte dos D elementos da query e da saída.
    // Supondo que cada thread processe os índices: c = tid, tid+blockSize, tid+2*blockSize, ...
    int numLocal = (D - tid + blockSize - 1) / blockSize;
    // Limite máximo (assumindo D razoável)
    const int MAX_LOCAL = 128;
    float acc[MAX_LOCAL];
    for (int i = 0; i < numLocal; i++) {
        acc[i] = 0.0f;
    }

    // Variáveis globais para acumulação da escala (softmax) – armazenadas em memória compartilhada
    __shared__ float m_shared;  // guarda o máximo atual dos scores (logits)
    __shared__ float l_shared;  // acumula a soma dos exp(scores - m)
    if(tid == 0) {
        m_shared = -INFINITY;
        l_shared = 0.0f;
    }
    __syncthreads();

    /*
      Reserva de memória compartilhada dinâmica:
        - Área para redução dos produtos parciais: array de tamanho (BC * blockSize)
        - Área para guardar os scores reduzidos de cada key do tile: array de tamanho (BC)
      Total de floats requeridos: BC * blockSize + BC.
    */
    extern __shared__ float s_mem[];
    float* smem = s_mem;                     // tamanho: BC * blockSize
    float* s_scores = smem + BC * blockSize;   // tamanho: BC

    // Número de tiles para percorrer os N keys
    int numTiles = (N + BC - 1) / BC;
    for (int tile = 0; tile < numTiles; tile++) {
        int key_start = tile * BC;
        int cur_BC = (key_start + BC <= N) ? BC : (N - key_start);

        // Cada thread computa, para cada key j do tile, um somatório parcial do produto dot(q, k_j)
        float partial[64];  // suposição: BC <= 64
        for (int j = 0; j < cur_BC; j++) {
            partial[j] = 0.0f;
        }
        // Loop sobre a dimensão D com step blockSize (dividindo entre as threads)
        for (int c = tid; c < D; c += blockSize) {
            float q_val = Q[qIdx * D + c];
            for (int j = 0; j < cur_BC; j++) {
                // Cada key j do tile: índice global = key_start + j
                float k_val = K[(key_start + j) * D + c];
                partial[j] += q_val * k_val;
            }
        }
        // Armazena os parciais em memória compartilhada para redução:
        // Cada thread escreve seus valores para cada key j na posição: [j * blockSize + tid]
        for (int j = 0; j < cur_BC; j++) {
            smem[j * blockSize + tid] = partial[j];
        }
        __syncthreads();

        // Redução: a thread 0 soma as contribuições de todas as threads para cada key j
        float s[64];  // scores finais para cada key j do tile
        if(tid == 0) {
            for (int j = 0; j < cur_BC; j++) {
                float sum = 0.0f;
                for (int i = 0; i < blockSize; i++) {
                    sum += smem[j * blockSize + i];
                }
                s[j] = sum;
            }
            // Copia os resultados para s_scores (para posterior broadcast)
            for (int j = 0; j < cur_BC; j++) {
                s_scores[j] = s[j];
            }
        }
        __syncthreads();

        // Calcular o máximo do tile (tile_max) a partir dos scores
        float tile_max = -INFINITY;
        if(tid == 0) {
            for (int j = 0; j < cur_BC; j++) {
                if(s_scores[j] > tile_max)
                    tile_max = s_scores[j];
            }
            // Armazena o tile_max na primeira posição de s_scores para broadcast
            s_scores[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_scores[0];  // todos os threads obtêm o mesmo valor

        // Atualiza o valor global do máximo usando m_shared
        float m_old = m_shared;
        float new_m = (m_old > tile_max) ? m_old : tile_max;

        // Cada thread calcula os expoentes para cada key do tile (exp(score - new_m))
        // A redução dos expoentes (tile_sum) é feita pela thread 0.
        float exp_vals[64];
        float tile_sum = 0.0f;
        if(tid < cur_BC) {
            exp_vals[tid] = expf(s_scores[tid] - new_m);
        }
        __syncthreads();
        // Thread 0 acumula a soma dos expoentes para este tile
        if(tid == 0) {
            for (int j = 0; j < cur_BC; j++) {
                tile_sum += exp_vals[j];
            }
        }
        __syncthreads();
        // Broadcast tile_sum para todos os threads (armazenando em s_scores[1], por exemplo)
        if(tid == 0) {
            s_scores[1] = tile_sum;
        }
        __syncthreads();
        tile_sum = s_scores[1];

        // Atualiza o acumulador global da soma (l_shared)
        // new_l = exp(m_old - new_m) * l_shared + tile_sum
        float new_l = expf(m_old - new_m) * l_shared + tile_sum;

        // Atualiza a saída acumulada (vetor de tamanho D) de forma distribuída entre as threads.
        // Cada thread processa os índices c = tid, tid+blockSize, ...
        for (int c = tid, idx = 0; c < D; c += blockSize, idx++) {
            float tile_acc = 0.0f;
            // Para cada key do tile, carregar o correspondente V e multiplicar pelo expoente calculado.
            for (int j = 0; j < cur_BC; j++) {
                float v_val = V[(key_start + j) * D + c];
                tile_acc += exp_vals[j] * v_val;
            }
            // Atualiza: escala o acumulado anterior e soma a nova contribuição
            acc[idx] = acc[idx] * expf(m_old - new_m) + tile_acc;
        }
        __syncthreads();
        // Apenas um thread atualiza os escalares globais para a query
        if(tid == 0) {
            m_shared = new_m;
            l_shared = new_l;
        }
        __syncthreads();
    } // fim do loop sobre tiles de keys

    // Normaliza a saída acumulada e grava em O
    float invL = 1.0f / l_shared;
    for (int c = tid, idx = 0; c < D; c += blockSize, idx++) {
        O[qIdx * D + c] = acc[idx] * invL;
    }
    // A thread 0 grava o log-sum-exp final para a query
    if(tid == 0) {
        L[qIdx] = m_shared + logf(l_shared);
    }
}

int main() {
    // Parâmetros – você pode ajustar conforme necessário
    static const int N  = 4096;   // número de queries/keys/values
    static const int D  = 4096;   // dimensão dos vetores
    static const int BC = 64;     // tile de keys
    size_t sz = (size_t)N * D;
    
    // Aloca memória na CPU
    float *h_Q = new float[sz];
    float *h_K = new float[sz];
    float *h_V = new float[sz];
    float *h_O = new float[sz];
    float *h_L = new float[N];
    
    // Aloca memória na GPU
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    checkCuda(cudaMalloc(&d_Q, sz * sizeof(float)));
    checkCuda(cudaMalloc(&d_K, sz * sizeof(float)));
    checkCuda(cudaMalloc(&d_V, sz * sizeof(float)));
    checkCuda(cudaMalloc(&d_O, sz * sizeof(float)));
    checkCuda(cudaMalloc(&d_L, N * sizeof(float)));
    
    // Inicializa Q, K, V com números aleatórios (usando CURAND)
    curandGenerator_t gen;
    checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    
    cudaEvent_t startRng, stopRng;
    checkCuda(cudaEventCreate(&startRng));
    checkCuda(cudaEventCreate(&stopRng));
    checkCuda(cudaEventRecord(startRng));
    checkCurand(curandGenerateUniform(gen, d_Q, sz));
    checkCurand(curandGenerateUniform(gen, d_K, sz));
    checkCurand(curandGenerateUniform(gen, d_V, sz));
    checkCuda(cudaEventRecord(stopRng));
    checkCuda(cudaEventSynchronize(stopRng));
    float msRng = 0;
    checkCuda(cudaEventElapsedTime(&msRng, startRng, stopRng));
    std::cout << "Tempo geração RNG: " << msRng / 1000.0f << " s\n";
    
    checkCurand(curandDestroyGenerator(gen));
    checkCuda(cudaEventDestroy(startRng));
    checkCuda(cudaEventDestroy(stopRng));

    // Lançamento do kernel otimizado
    // Usaremos um bloco por query (N blocos) e, por exemplo, 128 threads por bloco.
    const int threadsPerBlock = 128;
    dim3 gridDim(N);
    dim3 blockDim(threadsPerBlock);
    // Tamanho da memória compartilhada:
    // (BC * threadsPerBlock + BC) * sizeof(float)
    size_t sharedMemSize = (BC * threadsPerBlock + BC) * sizeof(float);
    
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));
    optimized_attention<<<gridDim, blockDim, sharedMemSize>>>(d_Q, d_K, d_V, d_O, d_L, N, D, BC);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSyAchronize(stop));
    float ms = 0;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Tempo GPU (kernel otimizado): " << ms / 1000.0f << " s\n";
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    
    // Copia os resultados para a CPU
    checkCuda(cudaMemcpy(h_O, d_O, sz * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_L, d_L, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Exibe alguns resultados para verificação
    std::cout << "Exemplo dos primeiros 16 valores da saída da query 0 (O[0,:]):\n";
    for (int c = 0; c < 16; c++) {
        std::cout << h_O[c] << " ";
    }
    std::cout << "\nL[0] = " << h_L[0] << "\n";
    
    // Libera memória
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_L;
    
    return 0;
}
