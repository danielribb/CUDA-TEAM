import torch
import flash_attention  
import time

batch_size = 4
num_heads = 64
seq_len = 4096
head_dim = 64
softmax_scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
causal = False
dropout_prob = 0.0

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")



start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
O_custom = flash_attention.fa2_forward(Q, K, V)
end_event.record()

torch.cuda.synchronize()

torch.cuda.synchronize()



scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale

if causal:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask == 1, float('-inf'))

attn_weights = torch.softmax(scores, dim=-1)

if dropout_prob > 0.0:
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_prob, training=True)

O_reference = torch.matmul(attn_weights, V)


elapsed_time = start_event.elapsed_time(end_event)
print("Tempo de execução do kernel CUDA: {:.3f} ms".format(elapsed_time))
if torch.allclose(O_custom, O_reference, atol=1e-4):
    print("\nOs resultados conferem (estão próximos)!")
else:
    print("\nOs resultados são diferentes!")
