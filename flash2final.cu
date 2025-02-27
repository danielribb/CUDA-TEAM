#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
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

inline void checkCurand(curandStatus_t err, const char* msg=nullptr) {
    if (err != CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND error";
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
        exit(1);
    }
}

__global__ void forward2(
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
        float new_m = rowMax > m_val ? rowMax : m_val;
        float rowSum = 0;
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

void cpu_attention(const float* Q, const float* K, const float* V, float* O, float* L, int N, int D) {
    for (int i = 0; i < N; i++) {
        float rowMax = -INFINITY;
        for (int j = 0; j < N; j++) {
            float s = 0;
            const float* qRow = Q + i*D;
            const float* kRow = K + j*D;
            for (int c = 0; c < D; c++) {
                s += qRow[c] * kRow[c];
            }
            if (s > rowMax) rowMax = s;
        }
        float sumExp = 0;
        for (int j = 0; j < N; j++) {
            float s = 0;
            const float* qRow = Q + i*D;
            const float* kRow = K + j*D;
            for (int c = 0; c < D; c++) {
                s += qRow[c] * kRow[c];
            }
            s = expf(s - rowMax);
            sumExp += s;
        }
        for (int c = 0; c < D; c++) {
            O[i*D + c] = 0;
        }
        for (int j = 0; j < N; j++) {
            float s = 0;
            const float* qRow = Q + i*D;
            const float* kRow = K + j*D;
            for (int c = 0; c < D; c++) {
                s += qRow[c] * kRow[c];
            }
            s = expf(s - rowMax) / sumExp;
            const float* vRow = V + j*D;
            for (int c = 0; c < D; c++) {
                O[i*D + c] += s * vRow[c];
            }
        }
        L[i] = rowMax + logf(sumExp);
    }
}

int main() {
    static const int N  = 4096;
    static const int D  = 4096;
    static const int BC = 64;
    size_t sz = (size_t)N * D;
    float *h_Q = new float[sz];
    float *h_K = new float[sz];
    float *h_V = new float[sz];
    float *h_O = new float[sz];
    float *h_L = new float[N];
    float *h_O_ref = new float[sz];
    float *h_L_ref = new float[N];
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    checkCuda(cudaMalloc(&d_Q, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_K, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_V, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_O, sz*sizeof(float)));
    checkCuda(cudaMalloc(&d_L, N*sizeof(float)));
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
    std::cout << "Tempo geracao RNG: " << msRng/1000.0f << " s\n";
    checkCurand(curandDestroyGenerator(gen));
    checkCuda(cudaEventDestroy(startRng));
    checkCuda(cudaEventDestroy(stopRng));
   // checkCuda(cudaMemcpy(h_Q, d_Q, sz*sizeof(float), cudaMemcpyDeviceToHost));
   // checkCuda(cudaMemcpy(h_K, d_K, sz*sizeof(float), cudaMemcpyDeviceToHost));
   // checkCuda(cudaMemcpy(h_V, d_V, sz*sizeof(float), cudaMemcpyDeviceToHost));
    //auto cpu_start = std::chrono::high_resolution_clock::now();
   // cpu_attention(h_Q, h_K, h_V, h_O_ref, h_L_ref, N, D);
    //auto cpu_end = std::chrono::high_resolution_clock::now();
   // std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
   // std::cout << "Tempo CPU: " << cpu_diff.count() << " s\n";
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));
    forward2<<<N, 32, D*sizeof(float)>>>(d_Q, d_K, d_V, d_O, d_L, N, D, BC);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float ms = 0;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Tempo GPU: " << ms/1000.0f << " s\n";
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaMemcpy(h_O, d_O, sz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_L, d_L, N*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Primeiros 16 valores CPU O[0,:]:\n";
  //  for (int c = 0; c < 32; c++) {
   //     std::cout << h_O_ref[c] << " ";
   // }
    std::cout << "\nCPU L[0] = " << h_L_ref[0] << "\n";
    std::cout << "Primeiros 16 valores GPU O[0,:]:\n";
    for (int c = 0; c < 800; c++) {
        std::cout << h_O[c] << " ";
    }
    std::cout << "\nGPU L[0] = " << h_L[0] << "\n";
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
    delete[] h_O_ref;
    delete[] h_L_ref;
    return 0;
}
