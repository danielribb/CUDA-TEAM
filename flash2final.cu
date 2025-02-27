#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

__global__ void flash_forward_2(float *q_g, float *k_g, float *v_g, float *o_g){

}




int main(){

    int seq_len = 512;
    int d_model = 1024;

    float *q_c, *k_c, *v_c, *o_c;
    float *q_g, *k_g, *v_g, *o_g;

    q_c = (float *)malloc(seq_len * d_model * sizeof(float));
    k_c = (float *)malloc(seq_len * d_model * sizeof(float));
    q_c = (float *)malloc(seq_len * d_model * sizeof(float));
    o_c = (float *)malloc(seq_len * d_model * sizeof(float));

    cudaMalloc(&q_g, seq_len * d_model * sizeof(float));
    cudaMalloc(&k_g, seq_len * d_model * sizeof(float));
    cudaMalloc(&v_g, seq_len * d_model * sizeof(float));
    cudaMalloc(&o_g, seq_len * d_model * sizeof(float));

    dim3 blockDim((101376 /(4 * d_model)), min((101376 /(4 * d_model)), d_model));
    dim3 gridDim((d_model + blockDim.x - 1)/blockDim.x, (seq_len + blockDim.y - 1)/blockDim.y);
    flash_forward_2<<<gridDim, blockDim>>>(q_g, k_g, v_g, o_g);
    cudaDeviceSynchronize();


}

