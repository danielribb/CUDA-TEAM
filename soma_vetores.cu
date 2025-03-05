#include <iostream>
using namespace std;

__global__ void Soma(int *a, int  *b, int *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) c[i] =  a[i] + b[i];
}


int main(){
    int n, x; 
    cin >> n;
    int *a = new int[n];
    int *b = new int[n];
    int *c = new int[n];

    for (int i = 0; i < n; i++) {
        cin >> x;
        a[i] = x;
    }
    for (int i = 0; i < n; i++) {
        cin >> x;
        b[i] = x;
    }

    int *pa, *pb, *pc;
    cudaMalloc(&pa, n*sizeof(int));
    cudaMalloc(&pb, n*sizeof(int));
    cudaMalloc(&pc, n*sizeof(int));
    cudaMemcpy(pa, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pb, b, n*sizeof(int), cudaMemcpyHostToDevice);
    
    int blocksize = 256;
    int gridsize = (n + blocksize - 1) / blocksize; 
    Soma<<<gridsize, blocksize>>>(pa, pb, pc, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, pc, n*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        cout << c[i] << " ";
    }

    cudaFree(pa);
    cudaFree(pb);
    cudaFree(pc);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
