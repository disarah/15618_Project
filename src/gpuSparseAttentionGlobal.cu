#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "gpuSparseAttentionGlobal.h"

#define TS 8
#define ID 16

__global__ void MultiHeadDDS(float* A, float* B, float* C, int M, int N, int K, int* indices, int I, int hv) {
    // A is MxK, B is NxK
    __shared__ float lA[ID][TS];
    __shared__ float lB[TS][ID];
    
    int h = threadIdx.x; // which head
    int q = threadIdx.y; // pos within tile
    int p = threadIdx.z;
    
    bool isV = (hv == 0);
    int i = (isV) ? gridDim.x * blockDim.y + threadIdx.y : indices[ blockIdx.y * I + threadIdx.y];
    int j = (isV) ? indices[ blockIdx.y * I + threadIdx.y] : gridDim.x * blockDim.y + threadIdx.y;

    float tmp = 0.0f;
    for(int c = 0; c < K / TS; c++){
        lA[q][p] = A[h * M * K + i * K + c * TS + p];
        lB[p][q] = B[h * K * N + (c * TS + p) * N + j];
        __syncthreads();
        for(int k = 0; k < TS; k++){
            tmp += lA[q][k] * lB[k][q];
        }
        __syncthreads();
    }
    C[h * M * N + i * N + j] = tmp;
}


__global__ void MultiHeadSoftMaxSparseGlobal(float* A, int M, int N, int* indices, int I, int full) {
    extern __shared__ float SM[];

    // full grid:   thread (NHEAD,  N, 1), block (ID, 1, 1)
    // columns:     thread (NHEAD, ID, 1), block ( N, 1, 1)
    int h = threadIdx.x; // which head
    bool isFull = (full == 1);
    int idxIdx = (!isFull) ? threadIdx.y : blockIdx.x;
    int globalCol = (!isFull) ? indices[idxIdx] : threadIdx.y;
    int globalRow = (!isFull) ? blockIdx.x : indices[idxIdx];
    int rowLen = (!isFull) ? I : N;


    SM[idxIdx] = A[h * M * N + globalRow * N + globalCol];

    __syncthreads();

    register float total = 0.0;

    for(int i = 0; i < rowLen; i++) {
        total += SM[i];
    }

    A[h * M * N + globalRow * N + globalCol] /= total;
}


__global__ void MultiHeadSDDGlobal(float* A, float* B, float* C, int M, int N, int K, int* indices, int I) {
    // A is MxK, B is NxK
    __shared__ float lA[TS][ID]; // (TS, #ind)
    __shared__ float lB[ID][TS]; // (#ind, TS)
    int i = blockDim.z * TS + threadIdx.z;
    int j = blockDim.y * TS + threadIdx.z;
    int rowLen = I;
    
    int h = threadIdx.x; // which head
    int q = threadIdx.y; // pos within tile
    int p = threadIdx.z;

    float tmp = 0.0f;

    int ind = indices[ blockIdx.x*I + threadIdx.y];
    lA[p][q] = A[h * M * K + i * K + ind];
    lB[q][p] = B[h * K * N + ind * N + j];
    __syncthreads();
    for(int k = 0; k < rowLen; k++){
        tmp += lA[p][k] * lB[k][p];
    }
    __syncthreads();
    C[h * M * N + i * N + j] = tmp;
}

__global__ void MultiHeadSDDGlobalFull(float* A, float* B, float* C, int M, int N, int K, int* indices, int I) {
    // A is MxK, B is NxK
    __shared__ float lA[ID][TS]; // (#ind, TS)
    __shared__ float lB[TS][ID]; // (TS, #ind)

    int i = indices[ blockIdx.x*I + threadIdx.y];
    int j = blockDim.y * TS + threadIdx.z;
    int rowLen = TS;
    
    int h = threadIdx.x; // which head
    int q = threadIdx.y; // pos within tile
    int p = threadIdx.z;

    float tmp = 0.0f;
    for(int c = 0; c < K / TS; c++){
        lA[q][p] = A[h * M * K + i * K + c * TS + p];
        lB[p][q] = B[h * K * N + (c * TS + p) * N + j];
        __syncthreads();
        for(int k = 0; k < rowLen; k++){
            tmp += lA[q][k] * lB[k][q];
        }
        __syncthreads();
    }
    
    C[h * M * N + i * N + j] = tmp;
}


void gpuSparseAttentionGlobal(int N, int D_MODEL, int N_HEAD) {
    int d_k = D_MODEL / N_HEAD;
    float sqrt_d_k = sqrt(d_k);
    int ws = 16; // window size needs to be a multiple of TS

    float *query, *key, *value, *attn_scores, *result;
    cudaMalloc((void **)&query, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&key, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&value, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&attn_scores, N_HEAD * N*N * sizeof(float));
    cudaMalloc((void **)&result, N_HEAD*N*d_k * sizeof(float));

    int numIdxs = ID; // numIdxs must be a multiple of N
    size_t idxSize = numIdxs*sizeof(int);
    int *idxs = (int*)malloc(idxSize);
    for(int i=0; i<numIdxs; i++){
        idxs[i] = i*ws;
    }
    int *indices;
    cudaMalloc((void **)&indices, idxSize);
    cudaMemcpy(indices, idxs, idxSize, cudaMemcpyHostToDevice);
    // =============================================================================================
    // Tiling + global sparse (horizontal then vertical pass)

    // dds
    dim3 threadPerBlock2(N_HEAD, min(TS, numIdxs), TS);
    int num_iterations = N/TS;
    dim3 numBlock2(num_iterations, max(1,numIdxs/TS), 1);

    // softmax
    dim3 threadPerBlock3(N_HEAD, numIdxs, 1);
    dim3 numBlock3(N, 1, 1);

    dim3 threadPerBlock4(N_HEAD, N, 1);
    dim3 numBlock4(numIdxs, 1, 1);

    // sdd
    dim3 threadPerBlock5(N_HEAD, min(TS, numIdxs), TS);
    dim3 numBlock5(max(1,numIdxs/TS), (d_k + TS-1)/TS, (N + TS-1)/TS);

    dim3 threadPerBlock6(N_HEAD, min(TS, numIdxs), TS);
    dim3 numBlock6(max(1,numIdxs/TS), (d_k + TS-1)/TS, 1);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    // query x key^T
    MultiHeadDDS<<<numBlock2, threadPerBlock2>>>(query, key, attn_scores, N, N, d_k, indices, numIdxs, 0);
    MultiHeadDDS<<<numBlock2, threadPerBlock2>>>(query, key, attn_scores, N, N, d_k, indices, numIdxs, 1);

    // softmax
    uint SMSize = sizeof(float) * numIdxs;
    MultiHeadSoftMaxSparseGlobal<<<numBlock3, threadPerBlock3, SMSize>>>(attn_scores, N, N, indices, numIdxs, 0);
    SMSize = sizeof(float) * N;
    MultiHeadSoftMaxSparseGlobal<<<numBlock4, threadPerBlock4, SMSize>>>(attn_scores, N, N, indices, numIdxs, 1);

    // attn x value int K, int* indices, int I, int hv) {
    MultiHeadSDDGlobal<<<numBlock5, threadPerBlock5>>>(attn_scores, value, result, N, d_k, N, indices, numIdxs);
    MultiHeadSDDGlobalFull<<<numBlock6, threadPerBlock6>>>(attn_scores, value, result, N, d_k, N, indices, numIdxs);

    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("gpu global attention (tile parallelization using shared mem): %f microseconds\n\n" ,elapsedTime*1000);

}