#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>

// Declarações das funções definidas em attention_cuda.cu
std::vector<torch::Tensor> attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
);

std::vector<torch::Tensor> attention_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor M,
    torch::Tensor dO,
    bool causal,
    float softmax_scale
);

// Ligação com PyBind11
PYBIND11_MODULE(custom_attention, m) {
    m.def("attention_forward", &attention_forward, "Attention forward (CUDA)");
    m.def("attention_backward", &attention_backward, "Attention backward (CUDA)");
}
