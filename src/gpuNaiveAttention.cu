#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "gpuNaiveAttention.h"

#define TS 8

__global__ void MultiHeadGEMM(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float lA[TS][TS];
    __shared__ float lB[TS][TS];
    
    int h = threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.z * blockIdx.z + threadIdx.z;
    int q = threadIdx.y;  // pos within tile
    int p = threadIdx.z;

    // if (i >= M || j >= N) return;
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


__global__ void naiveAttention(float* query, float* key, float* value,
                                int N, int N_HEAD, int d_k, float sqrt_d_k,
                                float* attn_scores, float* result) {

    int h = threadIdx.x;     // local pixel x index in this block
    float score;

    // query x key^T
    for(int n1 = 0; n1 < N; n1++) {
        for(int n2 = 0; n2 < N; n2++) {
            score = 0.0;
            for(int d=0; d < d_k; d++) {
                score += query[h*N*d_k + n1*d_k + d] * key[h*N*d_k + n2*d_k + d];
            }
            attn_scores[h*N*N + n1*N + n2] = score / sqrt_d_k;
        }
    }

    for(int n1 = 0; n1 < N; n1++) {
        float sum = 0.0;
        for(int n2 = 0; n2 < N; n2++) {
            sum += attn_scores[h*N*N + n1*N + n2];
        }
        for(int n2 = 0; n2 < N; n2++) {
            attn_scores[h*N*N + n1*N + n2] /= sum;
        }
    }

    for(int n1 = 0; n1 < N; n1++) {
        for(int d = 0; d < d_k; d++) {
            float sum = 0.0;
            for(int n2 = 0; n2 < N; n2++) {
                sum += attn_scores[h*N*N + n1*N + n2] * value[h*N*d_k + n2*d_k + d];
            }
            result[h*N*d_k + n1*d_k + d] = sum;
        }
    }
}


void gpuNaiveAttention(int N, int D_MODEL, int N_HEAD) {
    int d_k = D_MODEL / N_HEAD;
    float sqrt_d_k = sqrt(d_k);

    float *query, *key, *value, *attn_scores, *result;
    cudaMalloc((void **)&query, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&key, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&value, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&attn_scores, N_HEAD * N*N * sizeof(float));
    cudaMalloc((void **)&result, N_HEAD*N*d_k * sizeof(float));

    // =============================================================================================
    // super simple version.
    // each thread takes care of calculation for a single "head" in multihead attention
    dim3 threadPerBlock(N_HEAD);
    dim3 numBlock(1);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    // auto beg = std::chrono::high_resolution_clock::now();
	naiveAttention<<<numBlock, threadPerBlock>>>(query, key, value, N, N_HEAD, d_k, sqrt_d_k, attn_scores, result);

    cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("gpu naive attention (inefficient per-head parallelization): %fms\n" ,elapsedTime);

    // =============================================================================================
    // shared mem optimized ver (tiling)
    dim3 threadPerBlock2(N_HEAD, TS, TS);
    dim3 numBlock2(1, (N + TS-1)/TS, (N + TS-1)/TS);

    dim3 threadPerBlock4(N_HEAD, TS, TS);
    dim3 numBlock4(1, (d_k + TS-1)/TS, (N + TS-1)/TS);

    cudaEvent_t start2, stop2;
    elapsedTime = 0.0;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2,0);

    // query x key^T
	MultiHeadGEMM<<<numBlock2, threadPerBlock2>>>(query, key, attn_scores, N, N, d_k);

    // TODO: softmax

    // softmaxed attn scores x value 
    MultiHeadGEMM<<<numBlock4, threadPerBlock4>>>(attn_scores, value, result, N, d_k, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsedTime, start2,stop2);
    printf("gpu naive attention (tile parallelization using shared mem): %fms\n\n" ,elapsedTime);

}
