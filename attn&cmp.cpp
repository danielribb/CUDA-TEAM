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

void attention_backward(const float* queries, size_t num_queries, size_t dim,
                        const float* keys, size_t num_keys,
                        const float* values, size_t value_dim,
                        const float* grad_output,  // shape: num_queries x value_dim
                        float* grad_queries,       // shape: num_queries x dim
                        float* grad_keys,          // shape: num_keys x dim
                        float* grad_values) {      // shape: num_keys x value_dim
    for (size_t i = 0; i < num_queries; ++i) {
        // Ponteiro para a query atual
        const float* query = queries + i * dim;
        float scale = std::sqrt(static_cast<float>(dim));

        // 1. Recalcula os scores para a query atual.
        float* scores = new float[num_keys];
        for (size_t j = 0; j < num_keys; ++j) {
            const float* key = keys + j * dim;
            scores[j] = dotProduct(query, key, dim) / scale;
        }
        
        // 2. Aplica softmax para obter os pesos de atenção.
        float* weights = softmax(scores, num_keys);
        
        // 3. Computa sum_grad = sum_{l}(weights[l] * dot(grad_output_i, value_l))
        float sum_grad = 0.0f;
        for (size_t l = 0; l < num_keys; ++l) {
            const float* value = values + l * value_dim;
            float dot_val = 0.0f;
            for (size_t k = 0; k < value_dim; ++k) {
                dot_val += grad_output[i * value_dim + k] * value[k];
            }
            sum_grad += weights[l] * dot_val;
        }
        
        // 4. Calcula o gradiente dos scores: grad_z[j] = weights[j] * (dot(grad_output, value_j) - sum_grad)
        float* grad_z = new float[num_keys];
        for (size_t j = 0; j < num_keys; ++j) {
            const float* value = values + j * value_dim;
            float dot_val = 0.0f;
            for (size_t k = 0; k < value_dim; ++k) {
                dot_val += grad_output[i * value_dim + k] * value[k];
            }
            grad_z[j] = weights[j] * (dot_val - sum_grad);
        }
        
        // 5. Atualiza os gradientes para queries e keys.
        for (size_t j = 0; j < num_keys; ++j) {
            const float* key = keys + j * dim;
            for (size_t d = 0; d < dim; ++d) {
                // Gradiente em relação à query: acumula (grad_z * key) / scale
                grad_queries[i * dim + d] += grad_z[j] * key[d] / scale;
                // Gradiente em relação à key: acumula (grad_z * query) / scale
                grad_keys[j * dim + d] += grad_z[j] * query[d] / scale;
            }
        }
        
        // 6. Atualiza o gradiente para values: para cada key j, grad_values[j] += weights[j] * grad_output[i]
        for (size_t j = 0; j < num_keys; ++j) {
            for (size_t k = 0; k < value_dim; ++k) {
                grad_values[j * value_dim + k] += weights[j] * grad_output[i * value_dim + k];
            }
        }
        
        // Libera a memória alocada dinamicamente.
        delete[] scores;
        delete[] weights;
        delete[] grad_z;
    }
}

bool compareMatrices(const float* matA, const float* matB, size_t rows, size_t cols, float tolerance = 1e-6f) {
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

    float grad_queries[num_queries * dim] = {0.0f};
    float grad_keys[num_keys * dim] = {0.0f};
    float grad_values[num_keys * value_dim] = {0.0f};

    // Chama a função backward.
    attention_backward(queries, num_queries, dim,
                       keys, num_keys,
                       values, value_dim,
                       grad_output,
                       grad_queries, grad_keys, grad_values);

    // Exibe os gradientes das queries.
    std::cout << "Gradiente das queries:" << std::endl;
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            std::cout << grad_queries[i * dim + d] << " ";
        }
        std::cout << std::endl;
    }

    // Exibe os gradientes das keys.
    std::cout << "Gradiente das keys:" << std::endl;
    for (size_t j = 0; j < num_keys; ++j) {
        for (size_t d = 0; d < dim; ++d) {
            std::cout << grad_keys[j * dim + d] << " ";
        }
        std::cout << std::endl;
    }

    // Exibe os gradientes dos values.
    std::cout << "Gradiente dos values:" << std::endl;
    for (size_t j = 0; j < num_keys; ++j) {
        for (size_t k = 0; k < value_dim; ++k) {
            std::cout << grad_values[j * value_dim + k] << " ";
        }
       

    return 0;
}

//for collab
// !nvcc -o compare compare.cpp -arch=sm_75
// !./compare
