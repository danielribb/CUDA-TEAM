// attention and matrix compare function implemented on c++ for correctness check purposes
// %%writefile compare.cpp (needed for collab)
#include <iostream>
#include <cmath>

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

bool compareMatrices(const float* matA, const float* matB, size_t rows, size_t cols, float tolerance = 1e-5f) {
    size_t totalElements = rows * cols;
    for (size_t i = 0; i < totalElements; ++i) {
        if (std::fabs(matA[i] - matB[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // queries, keys, values, and dims not defined, supposed to use the ones from the code to be compared with
    float output[num_queries * value_dim];

    attention(queries, num_queries, dim, keys, num_keys, values, value_dim, output);

    for (size_t i = 0; i < num_queries; ++i) {
        std::cout << "Resultado da atenção para a query " << i + 1 << ": ";
        for (size_t j = 0; j < value_dim; ++j) {
            std::cout << output[i * value_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    if (compareMatrices(matrix1, matrix2, rows, cols)) {
        std::cout << "matrix1 e matrix2 são iguais." << std::endl;
    } else {
        std::cout << "matrix1 e matrix2 são diferentes." << std::endl;
    }

    return 0;
}

//for collab
// !nvcc -o compare compare.cpp -arch=sm_75
// !./compare
