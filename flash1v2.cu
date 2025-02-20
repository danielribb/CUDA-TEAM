#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void forward_kernel(const float* query_matrix_device_pointer, const float* key_matrix_device_pointer, const float* value_matrix_device_pointer, const int sequence_length, const int embedding_dimension,
                    const int total_columns_in_blocks, const int total_rows_in_blocks, const int block_size_columns, const int block_size_rows, const float softmax_scale,
                    float* sum_matrix_device_pointer, float *max_matrix_device_pointer, float* output_matrix_device_pointer) {
    int thread_index_x = threadIdx.x;
    int block_index_x = blockIdx.x; 
    int block_index_y = blockIdx.y;

    int qkv_offset = (block_index_x * gridDim.y * sequence_length * embedding_dimension) + (block_index_y * sequence_length * embedding_dimension); 
    int lm_offset = (block_index_x * gridDim.y * sequence_length) + (block_index_y * sequence_length);  

    extern __shared__ float shared_memory[];
    int tile_size = block_size_columns * embedding_dimension;  
    float* query_matrix_tile = shared_memory;
    float* key_matrix_tile = &shared_memory[tile_size];
    float* value_matrix_tile = &shared_memory[tile_size * 2];
    float* score_matrix_tile = &shared_memory[tile_size * 3];
    float eps = 1e-10;

    for (int column_block_index = 0; column_block_index < total_columns_in_blocks; column_block_index++) {
        for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
            key_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = key_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            value_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = value_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
        }
        __syncthreads();  

        for (int row_block_index = 0; row_block_index < total_rows_in_blocks; row_block_index++)  {
            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = query_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            }
            float row_max_previous = max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];
            float row_sum_previous = sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];

            float row_max = -INFINITY;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                float sum = 0;
                for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                    sum += query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] * key_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index];
                }
                sum *= softmax_scale;
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = sum;

                if (sum > row_max)
                    row_max = sum;
            }

            float row_sum = 0;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = expf(score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] - row_max);
                row_sum += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner];
            }

            float row_max_new = fmax(row_max_previous, row_max);
            float row_sum_new = (expf(row_max_previous - row_max_new) * row_sum_previous) + (expf(row_max - row_max_new) * row_sum);

            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                float probability_times_value = 0;  
                for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                    probability_times_value += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] * value_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index] + eps;
                }
                output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index] = (1 / (eps + row_sum_new)) \
                    * ((row_sum_previous * expf(row_max_previous - row_max_new) * output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index]) \
                    + (expf(row_max - row_max_new + eps) * probability_times_value));
            }
            max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_max_new;
            sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_sum_new;
        }
        __syncthreads();  
    }
}

template <typename T>
T* allocateDeviceMemory(size_t size) {
    T* device_ptr;
    cudaMalloc(&device_ptr, size);
    return device_ptr;
}

template <typename T>
void initializeMatrix(T* matrix, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);

    for (size_t i = 0; i < size; ++i) {
        matrix[i] = dis(gen);
    }
}

template <typename T>
void writeMatrixToFile(T* matrix, const std::string& filename, int batch_size, int num_heads, int sequence_length, int embedding_dimension) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < sequence_length; ++i) {
                for (int j = 0; j < embedding_dimension; ++j) {
                    file << matrix[(b * num_heads * sequence_length * embedding_dimension) +
                                   (h * sequence_length * embedding_dimension) +
                                   (i * embedding_dimension) + j];
                    if (j < embedding_dimension - 1) {
                        file << ", "; 
                    }
                }
                file << std::endl;
            }
            file << std::endl; 
        }
    }
    file.close();
}

template <typename T>
void printMatrix(T* matrix, int batch_size, int num_heads, int sequence_length, int embedding_dimension, int rowsToPrint, int colsToPrint) {
    T* host_matrix = new T[batch_size * num_heads * sequence_length * embedding_dimension];
    cudaMemcpy(host_matrix, matrix, batch_size * num_heads * sequence_length * embedding_dimension * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "Matrix:\n";
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < rowsToPrint; ++i) {
                for (int j = 0; j < colsToPrint; ++j) {
                    std::cout << host_matrix[(b * num_heads * sequence_length * embedding_dimension) +
                                            (h * sequence_length * embedding_dimension) +
                                            (i * embedding_dimension) + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    delete[] host_matrix;
}

int main() {
    const int batch_size = 1;
    const int num_heads = 1;
    const int sequence_length = 2;
    const int embedding_dimension = 2;
    float negative_infinity_host = -INFINITY;

    const int block_size_columns = 1;
    const int block_size_rows = 1;

    const int total_columns_in_blocks = ceil((float)sequence_length / block_size_columns);
    const int total_rows_in_blocks = ceil((float)sequence_length / block_size_rows);
    const float softmax_scale = 1.0f / sqrtf(embedding_dimension);

    size_t matrix_size = batch_size * num_heads * sequence_length * embedding_dimension * sizeof(float);
    size_t vector_size = batch_size * num_heads * sequence_length * sizeof(float);

    float* query_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* key_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* value_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];

    //initializeMatrix(query_matrix_host, batch_size * num_heads * sequence_length * embedding_dimension);
    //initializeMatrix(key_matrix_host, batch_size * num_heads * sequence_length * embedding_dimension);
    //initializeMatrix(value_matrix_host, batch_size * num_heads * sequence_length * embedding_dimension);

    query_matrix_host[0] = 1.0;
    query_matrix_host[1] = 2.0;
    query_matrix_host[2] = 3.0;
    query_matrix_host[3] = 4.0;

    key_matrix_host[0] = 4.0;
    key_matrix_host[1] = 3.0;
    key_matrix_host[2] = 2.0;
    key_matrix_host[3] = 1.0;

    value_matrix_host[0] = 2.0;
    value_matrix_host[1] = 1.0;
    value_matrix_host[2] = 4.0;
    value_matrix_host[3] = 3.0;

    float* query_matrix_device = allocateDeviceMemory<float>(matrix_size);
    float* key_matrix_device = allocateDeviceMemory<float>(matrix_size);
    float* value_matrix_device = allocateDeviceMemory<float>(matrix_size);
    float* output_matrix_device = allocateDeviceMemory<float>(matrix_size);
    float* sum_matrix_device = allocateDeviceMemory<float>(vector_size);
    float* max_matrix_device = allocateDeviceMemory<float>(vector_size);

    cudaMemcpy(query_matrix_device, query_matrix_host, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(key_matrix_device, key_matrix_host, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(value_matrix_device, value_matrix_host, matrix_size, cudaMemcpyHostToDevice);
    cudaMemset(output_matrix_device, 0, matrix_size);
    cudaMemset(sum_matrix_device, 0, vector_size);
    cudaMemset(max_matrix_device, *reinterpret_cast<int*>(&negative_infinity_host), vector_size); 

    const int shared_memory_size = (4 * block_size_columns * embedding_dimension * sizeof(float)) +
                                   (block_size_columns * block_size_rows * sizeof(float));

    dim3 grid_dim(batch_size, num_heads);
    dim3 block_dim(block_size_columns);

    forward_kernel<<<grid_dim, block_dim, shared_memory_size>>>(
        query_matrix_device, key_matrix_device, value_matrix_device, sequence_length,
        embedding_dimension, total_columns_in_blocks, total_rows_in_blocks, block_size_columns,
        block_size_rows, softmax_scale, sum_matrix_device, max_matrix_device, output_matrix_device);

    cudaDeviceSynchronize();

    float* output_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    cudaMemcpy(output_matrix_host, output_matrix_device, matrix_size, cudaMemcpyDeviceToHost);

    writeMatrixToFile(query_matrix_host, "query_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(key_matrix_host, "key_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(value_matrix_host, "value_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(output_matrix_host, "output_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);

    int rowsToPrint = sequence_length;
    int colsToPrint = embedding_dimension;
    std::cout << "Q:\n";
    printMatrix(query_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "K:\n";
    printMatrix(key_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "V:\n";
    printMatrix(value_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "O:\n";
    printMatrix(output_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);

    cudaFree(query_matrix_device);
    cudaFree(key_matrix_device);
    cudaFree(value_matrix_device);
    cudaFree(output_matrix_device);
    cudaFree(sum_matrix_device);
    cudaFree(max_matrix_device);

    delete[] query_matrix_host;
    delete[] key_matrix_host;
    delete[] value_matrix_host;
    delete[] output_matrix_host;

    return 0;
}
