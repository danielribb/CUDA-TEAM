#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <torch/types.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define PI 3.1415


__device__ __forceinline__ float warpReduceSum(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__device__ __forceinline__ float warpReduceMax(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}


template <const int Br, const int Bc>
__global__ void flash_attn_2_kernel(
    float* Q, float* K, float* V, int N, int d, 
    int Tr, int Tc, 
    float scale, float* L, float* O) {
    int tx = threadIdx.x; 

    int bx = blockIdx.x;  
    int by = blockIdx.y; 

    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_off = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * d;
    float* Sij = Vj + Bc * d;
    float* Oi = Sij + Br * Bc;
    float* li = Oi + Br * d;
    float* mi = li + Br;
    float* mi_new = mi + Br;

    int loads_per_thread_Brd = CEIL_DIV(d, Bc);
    int loads_per_thread_Bcd = CEIL_DIV(d, Br);

    for (int i = 0; i < Tr; i++) {
        for (int e = 0; e < loads_per_thread_Brd; e++) {
            int idx = e * (Br * Bc) + tx;
            int row = idx / d;
            if (idx < Br * d && i * Br + row < N) {
                int col = idx % d;
                Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
                Oi[row * d + col] = 0.0f;
            }
        }

        int s_row = tx / Bc;
        int s_col = tx % Bc;

        int global_row = (i * Br) + s_row;

        if (s_col == 0) {
            li[s_row] = 0.f;
            mi[s_row] = -INFINITY;
            mi_new[s_row] = -INFINITY;
        }
        __syncthreads();

        for (int j = 0; j < Tc; j++) {
            for (int e = 0; e < loads_per_thread_Bcd; e++) {
                int idx = e * (Br * Bc) + tx;
                int row = idx / d;
                if (idx < Bc * d && j * Bc + row < N) {
                    int col = idx % d;
                    Kj[row * d + col] = K[qkv_off + (j * Bc + row) * d + col];
                    Vj[row * d + col] = V[qkv_off + (j * Bc + row) * d + col];
                }
            }
            __syncthreads();


            float acc = 0.f;
            for (int k = 0; k < d; k++)
                acc += Qi[s_row * d + k] * Kj[s_col * d + k];

            acc *= scale;
            Sij[s_row * Bc + s_col] = acc;


            if (s_col == 0) {
                mi[s_row] = mi_new[s_row];
                float row_m = -INFINITY, row_l = 0.f;
                for (int c = 0; c < Bc; c++) {
                    float val = Sij[s_row * Bc + c];
                    if (val > row_m) {
                        row_m = val;
                    }
                }
                float maxval = max(mi[s_row], row_m);

                float kahan_comp = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    float exp_val = expf(Sij[s_row * Bc + c] - maxval);
                    Sij[s_row * Bc + c] = exp_val;

                    float y = exp_val - kahan_comp;  
                    float t = row_l + y;             
                    kahan_comp = (t - row_l) - y;   
                    row_l = t;
                }

                mi_new[s_row] = maxval;
                li[s_row] = expf(mi[s_row] - maxval) * li[s_row] + row_l;
            }
            __syncthreads();

            for (int col = s_col; col < d; col += Bc) {
                float acc = 0.f;
                float kahan_comp = 0.f;
                for (int c = 0; c < Bc; c++) {
                    float term = Sij[s_row * Bc + c] * Vj[c * d + col];

                    float y = term - kahan_comp;
                    float t = acc + y;
                    kahan_comp = (t - acc) - y;
                    acc = t;
                }

                Oi[s_row * d + col] = (mi[s_row] == -INFINITY || mi_new[s_row] == -INFINITY) ? acc : expf(mi[s_row] - mi_new[s_row]) * Oi[s_row * d + col] + acc;
            }
        }

        for (int col = s_col; col < d; col += Bc) {
            if (global_row < N)
                O[qkv_off + global_row * d + col] = (1 / li[s_row]) * Oi[s_row * d + col];
        }

        if (s_col == 0) {
            L[lm_off + (i * Br) + s_row] = mi_new[s_row] + logf(li[s_row]);
        }
        __syncthreads();
    }
}

torch::Tensor fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32;
    const int Br = 32;

    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    int Tc = ceil((float)N / Bc);
    int Tr = ceil((float)N / Br);
    float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});
    torch::Device device(torch::kCUDA);
    L = L.to(device);

    const int smem_size = ((Br * Bc) + (2 * Br * d) + (2 * Bc * d) + (3 * Br)) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

    dim3 grid_size(B, nh);     
    dim3 block_size(Br * Bc);  

    flash_attn_2_kernel<Br, Bc><<<grid_size, block_size, smem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tr, Tc, softmax_scale,
        L.data_ptr<float>(), O.data_ptr<float>());
    return O;
}

