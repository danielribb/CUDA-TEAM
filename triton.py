!pip install triton

import triton
import torch
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_L,
    stride_C,
    SEQ_LEN: tl.constexpr,
    D_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    block_index_q = tl.program_id(0)

    q_offset = block_index_q * BLOCK_SIZE_Q

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(SEQ_LEN, D_DIM),
        strides=(stride_L, stride_C),
        offsets=(q_offset, 0),
        block_shape=(BLOCK_SIZE_Q, D_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(SEQ_LEN, D_DIM),
        strides=(stride_L, stride_C),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, D_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(D_DIM, SEQ_LEN),
        strides=(stride_C, stride_L),
        offsets=(0, 0),
        block_shape=(D_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O,
        shape=(SEQ_LEN, D_DIM),
        strides=(stride_L, stride_C),
        offsets=(q_offset, 0),
        block_shape=(BLOCK_SIZE_Q, D_DIM),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, D_DIM], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr)
    for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        K_block = tl.load(K_block_ptr)
        V_block = tl.load(V_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
        QK_block = QK_block * softmax_scale - m_ij[:, None]
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block += tl.dot(P_block, V_block)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    m_i += tl.math.log(l_i)  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


class TritonAttention:
    @staticmethod
    def forward(Q, K, V, softmax_scale):
        D_DIM_Q, D_DIM_K = Q.shape[-1], K.shape[-1]
        D_DIM_V = V.shape[-1]
        assert D_DIM_Q == D_DIM_K and D_DIM_Q == D_DIM_V
        O = torch.empty_like(Q)
        SEQ_LEN = Q.shape[0]
        BLOCK_SIZE_Q = 64  # Defina o tamanho do bloco
        BLOCK_SIZE_KV = 64  # Defina o tamanho do bloco
        grid = lambda META: (triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q), 1)

        M = torch.empty((SEQ_LEN), dtype=torch.float32, device='cuda')
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_L=Q.stride(0),
            stride_C=Q.stride(1),
            SEQ_LEN=SEQ_LEN,
            D_DIM=D_DIM_K,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        )
        return O


def test_op(SEQ_LEN, D_DIM, dtype=torch.float16):
    # Gerando dados de entrada
    Q = torch.empty((SEQ_LEN, D_DIM), dtype=dtype, device='cuda').normal_(mean=0.0, std=0.5)
    K = torch.empty((SEQ_LEN, D_DIM), dtype=dtype, device='cuda').normal_(mean=0.0, std=0.5)
    V = torch.empty((SEQ_LEN, D_DIM), dtype=dtype, device='cuda').normal_(mean=0.0, std=0.5)
    softmax_scale = 1 / (D_DIM**0.5)


    # Implementação Flash Attention (Triton)
    tri_out = TritonAttention.forward(Q, K, V, softmax_scale)



    print("\nResultado da implementação Flash Attention (Triton):")
    print(tri_out)

  
# Testando a função
test_op(SEQ_LEN=1024, D_DIM=512)
