#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "gpuNaiveAttention.h"


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
    printf("gpu naive attention: %fms\n" ,elapsedTime);
    // std::cout << "gpu sparse attention: " << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << std::endl;

}
