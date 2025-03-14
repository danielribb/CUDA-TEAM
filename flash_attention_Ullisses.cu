#include<stdio.h>
#include<cuda_runtime.h>
#include<cmath>

__global__ void flash_attention(float *q, float *k, float *v,float *o, int seq_len, int head_dim, int *L, int block_size){
    __shared__ float q_block[8192];
    __shared__ float k_block[8192];
    __shared__ float v_block[8192];
    __shared__ float o_block[8192];
    __shared__ float l[64];
    __shared__ float m[64];
    __shared__ float m_intermediario[64];
    __shared__ float qk_block[4092];
    __shared__ float redu[4092];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    
    for(int i = 0; i < head_dim; i += blockDim.x){
        if(row < seq_len){
            q_block[row * head_dim + tx + i] = q[row * head_dim + tx + i];
            o_block[row * head_dim + tx + i] = 0;
        }
    }
    if(row < seq_len){
        l[ty] = 1.0;
        m[ty] = -INF;
    }
    float temp;
    for(int i = 0; i < head_dim; += blockDim.y){

        //carregando k e v
        for(int j =0; j < head_dim; j += blockDim.x){

            if(ty + i < seq_len){

                k_block[(tx + j)*blockDim.y + ty] = k[(ty + i) * head_dim + tx + j];

                v_block[ty * head_dim + tx + j] = v_block[(ty + i) * head_dim + tx + j ];

            }

        }
        //tiling para encontrar bloco kv
        for(int j = 0; j < block_size; j += blockDim.x){

            temp = 0;

            for(k = 0; k < head_dim; k++){

                temp = q_block[ty * head_dim + k] *k_block[ (tx + j) + block_size * k ];

            }

            qk_block[ty*head_dim + tx + j] = temp;

            redu = temp;

        }
        //redução para encontrar maior elemento
        int limite = head_dim/blockDim.y;
        for(int j = head_dim/2; j > 0; j = j/2){
    
            for(int k = 0; k < (j + blockDim.x - 1 )/blockDim.x; k++){
                
                if(tx + k < j){

                    if( redu[ty * block_dim + tx] < redu[ty * block_dim + tx + j] )

                        redu[ty * block_dim + tx] = redu[ty * block_dim + tx + j];

                }
            }

        }

        if(redu[ty * block_size] > m[ty])
            m_intermediario[ty] = redu[ty * block_size];


        
        for(int j = 0; j < block_size; j += blockDim.x){

            qk_block[ty * block_size + tx + j] =  exp(qk_block[ty * block_size + tx + j] - m[ty]);
            redu[ty * block_size + tx + j] = qk_block[ty * block_size + tx + j];

        }

        //reducao para calcular rowsum de p
        for(int j = head_dim/2; j > 0; j = j/2){
    
            for(int k = 0; k < (j + blockDim.x - 1 )/blockDim.x; k++){
                
                if(tx + k < j){

                    if( redu[ty * block_dim + tx] < redu[ty * block_dim + tx + j] )

                        redu[ty * block_dim + tx] += redu[ty * block_dim + tx + j];

                }
            }

        }

        // atualizando o valor de l
        l[ty] = exp(m[ty] - m_intermediario[ty]) + redu[ty];

        //tiling para calculo de qk * v

        for(int j = 0; j < block_size; j += blockdim.x){
            temp =0;
            for(int k = 0; k < blocksize; k++){

                temp += qk_block[ty * blocksize + k] * v_block[]

            }
            q_block[(ty + j) * ]
        }

        //calculando a matriz de atenção
        //qk_block é o produto PV
        o_block[ty * head_dim + tx] = exp(m[ty] - m_intermediario[ty]) * o[ty * head_dim + tx] + qk_block;

        m[ty] = m_intermediario[ty];

    }
    o_block[ty * head_dim + tx] = o[ty * head_dim + tx]*(1/l[ty]);

    l[ty] = m[ty] + log(l[ty]);
    L[row] = l[ty]
    for(int i = 0; i < head_dim; i+= blockDim.x){

        o[row + ty +  tx + i] = o_block[ty * head_dim + tx + i];

    }




}

