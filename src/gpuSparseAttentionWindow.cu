#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "gpuSparseAttentionWindow.h"

#define TS 8

__global__ void MultiHeadDDS(float* A, float* B, float* C, int M, int N, int K, int ws) {
    __shared__ float lA[TS][TS];
    __shared__ float lB[TS][TS];
    
    int h = threadIdx.x;
    int i = blockDim.x * ws + blockDim.z * blockIdx.z + threadIdx.z;
    int j = blockDim.x * ws +  blockDim.y * blockIdx.y + threadIdx.y;
    int q = threadIdx.y;  // pos within tile
    int p = threadIdx.z;

    float tmp = 0.0f;
    for(int c = 0; c < K / TS; c++){
        lA[p][q] = A[h * M * K + i * K + c * TS + q];
        lB[p][q] = B[h * K * N + (c * TS + p) * N + j];
        __syncthreads();
        for(int k = 0; k < TS; k++){
            tmp += lA[p][k] * lB[k][q];
        }
        __syncthreads();
    }
    C[h * M * N + i * N + j] = tmp;
}


__global__ void MultiHeadSoftMaxSparse(float* A, int M, int N, int ws) {
    extern __shared__ float SM[];

    int h = threadIdx.x;
    int localCol = threadIdx.y;
    int globalCol = blockDim.y * blockIdx.x + threadIdx.y;
    // int localRow = blockIdx.y;
    int globalRow = blockDim.y * blockIdx.x + blockIdx.y;

    // register float cnt = 0.0;
    // for(int i = 0; i <ws; i++) {
    //     cnt += A[h * M * N + globalRow * N + i];
    // }
    SM[localCol] = A[h * M * N + globalRow * N + globalCol];

    __syncthreads();

    register float total = 0.0;

    for(int i = 0; i < ws; i++) {
        total += SM[i];
    }

    A[h * M * N + globalRow * N + globalCol] /= total;
}


__global__ void MultiHeadSDD(float* A, float* B, float* C, int M, int N, int K, int ws) {
    __shared__ float lA[TS][TS];
    __shared__ float lB[TS][TS];
    
    int h = threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.z * blockIdx.z + threadIdx.z;
    int q = threadIdx.y;  // pos within tile
    int p = threadIdx.z;

    float tmp = 0.0f;
    int start = int(i/TS);
    int end = ws / TS;
    for(int c = start; c < end; c++){
        lA[p][q] = A[h * M * K + i * K + c * TS + q];
        lB[p][q] = B[h * K * N + (c * TS + p) * N + j];
        __syncthreads();
        for(int k = 0; k < TS; k++){
            tmp += lA[p][k] * lB[k][q];
        }
        __syncthreads();
    }
    C[h * M * N + i * N + j] = tmp;
}


__global__ void sparseAttention(float* query, float* key, float* value,
                                int N, int N_HEAD, int d_k, float sqrt_d_k,
                                float* attn_scores, float* result, int ws) {

    int h = threadIdx.x;     // local pixel x index in this block
    float score;

    // query x key^T
    for(int n1 = 0; n1 < N; n1++) {
        int start = int(n1 / ws) * ws;
        int end = start + ws;
        for(int n2 = start; n2 < end; n2++) {
            score = 0.0;
            for(int d=0; d < d_k; d++) {
                score += query[h*N*d_k + n1*d_k + d] * key[h*N*d_k + n2*d_k + d];
            }
            attn_scores[h*N*N + n1*N + n2] = score / sqrt_d_k;
        }
    }

    for(int n1 = 0; n1 < N; n1++) {
        float sum = 0.0;
        int start = int(n1 / ws) * ws;
        int end = start + ws;
        for(int n2 = start; n2 < end; n2++) {
            sum += attn_scores[h*N*N + n1*N + n2];
        }
        for(int n2 = start; n2 < end; n2++) {
            attn_scores[h*N*N + n1*N + n2] /= sum;
        }
    }

    for(int n1 = 0; n1 < N; n1++) {
        int start = int(n1 / ws) * ws;
        int end = start + ws;
        for(int d = 0; d < d_k; d++) {
            float sum = 0.0;
            for(int n2 = start; n2 < end; n2++) {
                sum += attn_scores[h*N*N + n1*N + n2] * value[h*N*d_k + n2*d_k + d];
            }
            result[h*N*d_k + n1*d_k + d] = sum;
        }
    }
}

void gpuSparseAttentionWindow(int N, int D_MODEL, int N_HEAD) {
    int d_k = D_MODEL / N_HEAD;
    float sqrt_d_k = sqrt(d_k);
    int ws = 16; // window size needs to be a multiple of TS

    float *query, *key, *value, *attn_scores, *result;
    cudaMalloc((void **)&query, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&key, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&value, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&attn_scores, N_HEAD * N*N * sizeof(float));
    cudaMalloc((void **)&result, N_HEAD*N*d_k * sizeof(float));

    // =============================================================================================
    // Super simple version with parallelization across heads
    dim3 threadPerBlock(N_HEAD);
    dim3 numBlock(1);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
	sparseAttention<<<numBlock, threadPerBlock>>>(query, key, value, N, N_HEAD, d_k, sqrt_d_k, attn_scores, result, ws);

    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("gpu sparse attention (inefficient per-head parallelization): %f microseconds\n" ,elapsedTime*1000);

    // =============================================================================================
    // Tiling + window sparse
    dim3 threadPerBlock2(N_HEAD, TS, TS);
    int num_windows = N/ws;
    dim3 numBlock2(num_windows, (ws+TS-1)/TS, (ws+TS-1)/TS);

    dim3 threadPerBlock3(N_HEAD, ws, 1);
    dim3 numBlock3(num_windows, ws, 1);

    dim3 threadPerBlock4(N_HEAD, TS, TS);
    dim3 numBlock4(1, (d_k + TS-1)/TS, (N + TS-1)/TS);

    cudaEvent_t start2, stop2;
    elapsedTime = 0.0;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2,0);

    // query x key^T
    MultiHeadDDS<<<numBlock2, threadPerBlock2>>>(query, key, attn_scores, N, N, d_k, ws);

    // softmax
    uint SMSize = sizeof(float) * ws;
    MultiHeadSoftMaxSparse<<<numBlock3, threadPerBlock3, SMSize>>>(attn_scores, N, N, ws);

    MultiHeadSDD<<<numBlock4, threadPerBlock4>>>(attn_scores, value, result, N, d_k, N, ws);

    cudaDeviceSynchronize();
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsedTime, start2,stop2);
    printf("gpu sparse attention (tile parallelization using shared mem): %f microseconds\n\n" ,elapsedTime*1000);
}
