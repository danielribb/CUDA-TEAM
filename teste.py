import torch
import torch.nn.functional as F
import flash_cuda  # Certifique-se de que o módulo compilado está disponível

def attention_ref(Q, K, V, causal, softmax_scale):
    """
    Implementação de referência para a atenção:
      - Calcula scores = Q @ K^T * softmax_scale.
      - Se causal, aplica máscara triangular inferior.
      - Aplica softmax e multiplica por V.
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        # Cria máscara causal: lower triangular (True onde o elemento deve ser mantido)
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, -1e9)
    P = F.softmax(scores, dim=-1)
    O = torch.matmul(P, V)
    return O

def main():
    # Parâmetros maiores
    batch_size = 8
    num_heads = 16
    seq_len = 4096   # Sequência longa
    head_dim = 64
    causal = True
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Cria tensores aleatórios para Q, K e V (float32 e no dispositivo CUDA)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    # Executa o forward da atenção usando a extensão CUDA
    O_ext, M_ext = flash_cuda.attention_forward(Q, K, V, causal, softmax_scale)
    
    # Calcula a saída de referência usando PyTorch
    O_ref = attention_ref(Q, K, V, causal, softmax_scale)
    
    # Compara as saídas
    error = (O_ext - O_ref).abs().max().item()
    print("Erro máximo absoluto:", error)
    print("Forma da saída (extensão):", O_ext.shape)
    print("Forma da saída (referência):", O_ref.shape)
    
    # Exibe alguns valores de exemplo para comparação
    print("Extensão O[0, 0, 0, :10]:", O_ext[0, 0, 0, :10])
    print("Referência O[0, 0, 0, :10]:", O_ref[0, 0, 0, :10])

if __name__ == "__main__":
    main()
