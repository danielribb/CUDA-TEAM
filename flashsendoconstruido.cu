#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
Key Principles:
Thread Utilization:Ensure that the number of threads per block is a multiple of warpSize (typically 32) to avoid underutilization of hardware resources.
Shared Memory Alignment:When using shared memory, BLOCK_SIZE or TILE_WIDTH should divide evenly into NUM_SAMPLES and FEATURE_DIMENSION to minimize boundary conditions and wasted computations.
Grid Size Optimization:Ensure the grid dimensions align with the block size so that each thread block processes a uniform amount of work.
Avoid Divergence:Avoid irregular boundary conditions by padding NUM_SAMPLES or FEATURE_DIMENSION to the nearest multiple of BLOCK_SIZE if they are not already divisible.
These adjustments will improve efficiency by minimizing divergence and maximizing thread utilization and shared memory bandwidth.
*/

#define BLOCK_SIZE 32
#define MEM_WIDTH 32
#define TILE_WIDTH 32

#define NUM_SAMPLES 8192
#define FEATURE_DIMENSION 8192

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line
            << " '" << func << "'\n";
        std::cerr << "Error: " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

void generateRandomMatrix(float* matrix, int rows, int cols) {
    srand(0);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int y = 0; y < rows; y+=400) {
        for (int x = 0; x < cols; x+=400) {
            printf("%.6f ", matrix[y * cols + x]);
        }
        printf("\n");
    }
    printf("\n");
}


void transposeMatrix(const float* inputMatrix, float* transposedMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposedMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

__global__ void shared_compute_scores(float* queryMatrix, float* keyTransposeMatrix, float* attentionScores) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int scoreColumnIndex = blockX * TILE_WIDTH + threadX;
    int scoreRowIndex = blockY * TILE_WIDTH + threadY;
    float scoreValue = 0.0f;

    int numPhases = (FEATURE_DIMENSION + TILE_WIDTH - 1) / TILE_WIDTH;

    __shared__ float sharedQuery[MEM_WIDTH][MEM_WIDTH];
    __shared__ float sharedKeyTranspose[MEM_WIDTH][MEM_WIDTH];

    for (int phase = 0; phase < numPhases; phase++) {
        if (phase * TILE_WIDTH + threadX < FEATURE_DIMENSION && blockY * TILE_WIDTH + threadY < NUM_SAMPLES) {
            sharedQuery[threadY][threadX] = queryMatrix[(blockY * TILE_WIDTH + threadY) * FEATURE_DIMENSION + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedQuery[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < FEATURE_DIMENSION && blockX * TILE_WIDTH + threadX < NUM_SAMPLES) {
            sharedKeyTranspose[threadY][threadX] = keyTransposeMatrix[(phase * TILE_WIDTH + threadY) * NUM_SAMPLES + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedKeyTranspose[threadY][threadX] = 0.0f;
        }
        __syncthreads();

        if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                scoreValue += sharedQuery[threadY][i] * sharedKeyTranspose[i][threadX];
            }
        }

        __syncthreads();
    }

    if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
        attentionScores[scoreRowIndex * NUM_SAMPLES + scoreColumnIndex] = scoreValue / sqrtf(static_cast<float>(FEATURE_DIMENSION));
    }
}

__global__ void shared_softmax(float* attentionScores, float* softmaxScores) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowIndex < NUM_SAMPLES) {
        float maxScore = -1e30f;

        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            maxScore = fmaxf(maxScore, attentionScores[rowIndex * NUM_SAMPLES + colIndex]);
        }

        float sumExp = 0.0f;

        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] = expf(attentionScores[rowIndex * NUM_SAMPLES + colIndex] - maxScore);
            sumExp += softmaxScores[rowIndex * NUM_SAMPLES + colIndex];
        }

        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] /= sumExp;
        }
    }
}

__global__ void shared_compute_output(float* softmaxScores, float* valueMatrix, float* outputMatrix) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int outputColumnIndex = blockX * TILE_WIDTH + threadX;
    int outputRowIndex = blockY * TILE_WIDTH + threadY;
    float outputValue = 0.0f;

    int numPhases = (NUM_SAMPLES + TILE_WIDTH - 1) / TILE_WIDTH;

    __shared__ float sharedSoftmaxScores[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedValueMatrix[TILE_WIDTH][TILE_WIDTH];

    for (int phase = 0; phase < numPhases; phase++) {
        if (phase * TILE_WIDTH + threadX < NUM_SAMPLES && blockY * TILE_WIDTH + threadY < NUM_SAMPLES) {
            sharedSoftmaxScores[threadY][threadX] = softmaxScores[(blockY * TILE_WIDTH + threadY) * NUM_SAMPLES + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedSoftmaxScores[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < NUM_SAMPLES && blockX * TILE_WIDTH + threadX < FEATURE_DIMENSION) {
            sharedValueMatrix[threadY][threadX] = valueMatrix[(phase * TILE_WIDTH + threadY) * FEATURE_DIMENSION + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedValueMatrix[threadY][threadX] = 0.0f;
        }

        __syncthreads();

        if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                outputValue += sharedSoftmaxScores[threadY][i] * sharedValueMatrix[i][threadX];
            }
        }

        __syncthreads();
    }

    if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
        outputMatrix[outputRowIndex * FEATURE_DIMENSION + outputColumnIndex] = outputValue;
    }
}

void computeAttentionGPUShared(float* queryMatrix, float* keyTransposeMatrix, float* valueMatrix, float* attentionScores, float* outputMatrix) {
    float* deviceQuery, * deviceKeyTranspose, * deviceValue, * deviceAttentionScores, * deviceSoftmaxScores, * deviceOutput;

    cudaMalloc(&deviceQuery, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceKeyTranspose, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceValue, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceAttentionScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    cudaMemcpy(deviceQuery, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKeyTranspose, keyTransposeMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValue, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDimension(TILE_WIDTH, TILE_WIDTH); 
    dim3 gridDimension((NUM_SAMPLES + blockDimension.x - 1) / blockDimension.x, (NUM_SAMPLES + blockDimension.y - 1) / blockDimension.y);

    shared_compute_scores << <gridDimension, blockDimension >> > (deviceQuery, deviceKeyTranspose, deviceAttentionScores);
    cudaDeviceSynchronize();

    dim3 softmaxBlockDimension(1, BLOCK_SIZE); 
    dim3 softmaxGridDimension(1, (NUM_SAMPLES + softmaxBlockDimension.y - 1) / softmaxBlockDimension.y);


    shared_softmax << <softmaxGridDimension, softmaxBlockDimension >> > (deviceAttentionScores, deviceSoftmaxScores);
    cudaDeviceSynchronize();

 
    dim3 outputBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 outputGrid((FEATURE_DIMENSION + outputBlock.x - 1) / outputBlock.x, (NUM_SAMPLES + outputBlock.y - 1) / outputBlock.y);
    shared_compute_output << <outputGrid, outputBlock >> > (deviceSoftmaxScores, deviceValue, deviceOutput);
    cudaDeviceSynchronize();

    cudaMemcpy(attentionScores, deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceQuery);
    cudaFree(deviceKeyTranspose);
    cudaFree(deviceValue);
    cudaFree(deviceAttentionScores);
    cudaFree(deviceSoftmaxScores);
    cudaFree(deviceOutput);
}

// -------------------------------------------------------------------------------------------------------------

int main() {
    float* queryMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* keyMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* valueMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* outputCPU = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* outputGPUGlobal = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* outputGPUShared = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* attentionScoresCPU = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* attentionScoresGlobal = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* attentionScoresShared = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* transposedKeyMatrix = new float[FEATURE_DIMENSION * NUM_SAMPLES];

    generateRandomMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    generateRandomMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    generateRandomMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    transposeMatrix(keyMatrix, transposedKeyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

    printMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Query Matrix");
    printMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Key Matrix");
    printMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Value Matrix");


    float gpu_shared_milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    computeAttentionGPUShared(queryMatrix, transposedKeyMatrix, valueMatrix, attentionScoresShared, outputGPUShared);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_shared_milliseconds, start, stop);
    printMatrix(attentionScoresShared, NUM_SAMPLES, NUM_SAMPLES, "GPU Shared Attention Scores");
    printMatrix(outputGPUShared, NUM_SAMPLES, FEATURE_DIMENSION, "GPU Shared Output Matrix");


    float gpu_shared_seconds = gpu_shared_milliseconds/=1000;
    printf("GPU Shared Execution Time: %.3f s\n", gpu_shared_seconds);

    delete[] queryMatrix;
    delete[] keyMatrix;
    delete[] valueMatrix;
    delete[] outputCPU;
    delete[] outputGPUGlobal;
    delete[] outputGPUShared;
    delete[] attentionScoresCPU;
    delete[] attentionScoresGlobal;
    delete[] attentionScoresShared;
    delete[] transposedKeyMatrix;

    return 0;
}
