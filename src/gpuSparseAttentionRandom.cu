#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <set>
#include<algorithm>

#include "gpuSparseAttentionRandom.h"

// #define TS 8

__global__ void MultiHeadDDS(float* A, float* B, float* C, int M, int N, int K, int* idx, int num_random) {
    int h = blockIdx.x;
    int i = blockDim.x * blockIdx.y + threadIdx.x;
    int tmp_j = i * num_random + threadIdx.y;
    if (tmp_j < N * num_random && tmp_j >= 0 && i < N) {
        int j = idx[i * num_random + threadIdx.y];
        float tmp = 0.0f;
        for(int k = 0; k < K; k++) {
            tmp += A[h * M * K + i * K + k] * B[h * K * N + k * N + j]; 
        }
        C[h * M * N + i * N + j] = tmp;
    }
    
}

__global__ void MultiHeadSoftMaxSparse(float* A, int M, int N, int* idx, int num_random) {
    extern __shared__ float SM[];

    int h = blockIdx.x;
    int localRow = threadIdx.x;
    int globalRow = blockDim.x * blockIdx.y + threadIdx.x;

    int tmpIdx = globalRow * num_random + threadIdx.y;
    if (tmpIdx < N * num_random && tmpIdx >= 0) {
        int globalCol = idx[tmpIdx];
        SM[localRow * num_random + globalCol] = A[h * M * N + globalRow * N + globalCol];
    }
    
    __syncthreads();

    register float total = 0.0;

    if (localRow < blockDim.x && tmpIdx < N * num_random && tmpIdx >= 0) {
        int start = localRow * num_random;
        for(int i = start; i < start + num_random; i++) {
            total += SM[i];
        }
        int globalCol = idx[tmpIdx];
        A[h * M * N + globalRow * N + globalCol] /= total;
    }
}



void gpuSparseAttentionRandom(int N, int D_MODEL, int N_HEAD, float RANDOM_FRAC) {
    int d_k = D_MODEL / N_HEAD;
    float sqrt_d_k = sqrt(d_k);

    float *query, *key, *value, *attn_scores, *result;
    cudaMalloc((void **)&query, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&key, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&value, N_HEAD*N*d_k * sizeof(float));
    cudaMalloc((void **)&attn_scores, N_HEAD * N*N * sizeof(float));
    cudaMalloc((void **)&result, N_HEAD*N*d_k * sizeof(float));

    // =============================================================================================
    // Tiling + random sparse

    // init random stuff
    int num_random = N * RANDOM_FRAC;
    std::vector<std::set<int>> random_idx_tmp;
    random_idx_tmp.resize(N);

    for (int i = 0; i < N; i++) {
        while(random_idx_tmp[i].size() != num_random) {
            int randomNumber = rand();
            random_idx_tmp[i].insert(randomNumber % N);
        }
    }

    int *random_idx;
    cudaMalloc((void **)&random_idx, N * num_random * sizeof(int));
    
    std::vector<std::vector<int>> random_idx_vec;
    random_idx_vec.resize(N);
    for (int i = 0; i < N; i++) {
        std::vector<int> l(random_idx_tmp[i].begin(), random_idx_tmp[i].end());
        std::sort(l.begin(), l.end());
        random_idx_vec[i] = l;
    }

    cudaMemcpy(random_idx, random_idx_vec.data(), sizeof(int) * N * num_random, cudaMemcpyHostToDevice);

    // block/thread size and timer 
    int rows_per_block = (1024 + num_random-1) / num_random;
    dim3 threadPerBlock1(rows_per_block, num_random);
    dim3 numBlock1(N_HEAD, (N+rows_per_block-1)/rows_per_block);

    uint SMSize = sizeof(float) * rows_per_block * num_random;
    dim3 threadPerBlock2(rows_per_block, num_random);
    dim3 numBlock2(N_HEAD, (N+rows_per_block-1)/rows_per_block);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    // Kernel Here
    MultiHeadDDS<<<numBlock1, threadPerBlock1>>>(query, key, attn_scores, N, N, d_k, random_idx, num_random);

    MultiHeadSoftMaxSparse<<<numBlock2, threadPerBlock2, SMSize>>>(attn_scores, N, N, random_idx, num_random);

    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("gpu random attention (tile parallelization using shared mem): %f microseconds\n\n" ,elapsedTime*1000);

}
