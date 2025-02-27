#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

inline void checkCuda(cudaError_t err, const char* msg=nullptr) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err);
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
        exit(1);
    }
}

__global__ void blockwise_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, int D, int BC
) {
    int r = blockIdx.x;
    if (r >= N) return;
    if (threadIdx.x != 0) return;
    const float* qRow = Q + r*D;
    float m_val = -INFINITY;
    float l_val = 0.0f;
    extern __shared__ float shmem[];
    float* o_val = shmem;
    for (int c = 0; c < D; c++) {
        o_val[c] = 0.0f;
    }
    int Tc = (N + BC - 1) / BC;
    for (int j = 0; j < Tc; j++) {
        int kvStart = j * BC;
        int curBc = (kvStart + BC <= N) ? BC : (N - kvStart);
        if (curBc <= 0) break;
        float rowMax = -INFINITY;
        float sVals[64];
        for (int b = 0; b < curBc; b++) {
            const float* kRow = K + (kvStart + b)*D;
            float sum = 0.0f;
            for (int c = 0; c < D; c++) {
                sum += qRow[c] * kRow[c];
            }
            sVals[b] = sum;
            if (sum > rowMax) rowMax = sum;
        }
        float new_m = (rowMax > m_val) ? rowMax : m_val;
        float rowSum = 0.0f;
        for (int b = 0; b < curBc; b++) {
            float p = expf(sVals[b] - new_m);
            sVals[b] = p;
            rowSum += p;
        }
        float l_new = expf(m_val - new_m)*l_val + rowSum;
        float alpha = expf(m_val - new_m);
        for (int c = 0; c < D; c++) {
            o_val[c] *= alpha;
        }
        for (int b = 0; b < curBc; b++) {
            float p = sVals[b];
            const float* vRow = V + (kvStart + b)*D;
            for (int c = 0; c < D; c++) {
                o_val[c] += p * vRow[c];
            }
        }
        m_val = new_m;
        l_val = l_new;
    }
    float invL = 1.0f / l_val;
    for (int c = 0; c < D; c++) {
        o_val[c] *= invL;
    }
    float logSumExp = m_val + logf(l_val);
    float* outRow = O + r*D;
    for (int c = 0; c < D; c++) {
        outRow[c] = o_val[c];
    }
    L[r] = logSumExp;
}

int main() {
    std::cout << "Iniciando blockwise_attention para 4096x4096...\n";
    static const int N  = 4096;
    static const int D  = 4096;
    static const int BC = 64;
    size_t sz = (size_t)N * D;
    float *h_Q = new float[sz];
    float *h_K = new float[sz];
    float *h_V = new float[sz];
    float *h_O = new float[sz];
    float *h_L = new float[N];
    for (size_t i = 0; i < sz; i++) {
        h_Q[i] = 1.0f + 0.00001f*(i % 104);
        h_K[i] = 2.0f + 0.00002f*(i % 89);
        h_V[i] = 3.0f + 0.00003f*(i % 125);
    }
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    checkCuda(cudaMalloc(&d_Q, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_K, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_V, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_O, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_L, N*sizeof(float)));
    checkCuda(cudaMemcpy(d_Q, h_Q, sz*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_K, h_K, sz*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_V, h_V, sz*sizeof(float), cudaMemcpyHostToDevice));
    dim3 grid(N, 1, 1);
    dim3 block(32, 1, 1);
    size_t shmSize = D * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();
    blockwise_attention_kernel<<<grid, block, shmSize>>>(d_Q, d_K, d_V, d_O, d_L, N, D, BC);
    checkCuda(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tempo de execucao do kernel: " << diff.count() << " s\n";

    checkCuda(cudaMemcpy(h_O, d_O, sz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_L, d_L, N*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Primeiros 8 valores de O[0,:]:\n";
    for (int c = 0; c < 8; c++) {
        std::cout << h_O[c] << " ";
    }
    std::cout << "\nL[0] = " << h_L[0] << "\n";
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
    std::cout << "Fim.\n";
    return 0;
}
