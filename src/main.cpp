#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <set>
#include<algorithm>


#include "gpuNaiveAttention.h"
#include "gpuSparseAttentionWindow.h"
#include "gpuSparseAttentionRandom.h"

#define N 1024  // multiple of 8
#define RANDOM_FRAC 0.015  // fraction of N per row for random sparse attn
#define D_MODEL 512 // multiple of 8
#define N_HEAD 8    // should be able to divide D_MODEL
#define d_k D_MODEL / N_HEAD

class CPUNaiveAttention {
  public:
    
    // float query[N_HEAD][N][d_k];
    // float key[N_HEAD][N][d_k];
    // float value[N_HEAD][N][d_k];

    float *query;
    float *key;
    float *value;


    CPUNaiveAttention() {
        query = (float*)malloc(N_HEAD*N*d_k * sizeof(float));
        key = (float*)malloc(N_HEAD*N*d_k* sizeof(float));
        value = (float*)malloc(N_HEAD*N*d_k* sizeof(float));
    }

    int get(int h, int n, int d) {return h*N*d_k + n*d_k + d;}   // Implicitly inline

    void sparse_scaled_dot_product_attention_window() {
        float *attn_scores = (float*)malloc(N_HEAD*N*N* sizeof(float));
        float *result = (float*)malloc(N_HEAD*N*d_k * sizeof(float));
        auto beg = std::chrono::steady_clock::now();

        // query x key^T
        int ws = 16;
        float sqrt_d_k = sqrt(d_k);
        float score;
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                int start = int(n1 / ws) * ws;
                int end = start + ws;
                for(int n2 = start; n2 < end; n2++) {
                    score = 0.0;
                    for(int d=0; d < d_k; d++) {
                        score += query[get(h,n1,d)] * key[get(h,n2,d)];
                    }
                    attn_scores[h*N*N + n1*N + n2] = score / sqrt_d_k;
                }
            }
        }

        // softmax on attn_scores
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                float sum = 0.0;
                int start = int(n1 / ws) * ws;
                int end = start + ws;
                for(int n2 = start; n2 < end; n2++) {
                    sum += attn_scores[h*N*N + n1*N + n2];
                }
                for(int n2 = 0; n2 < N; n2++) {
                    attn_scores[h*N*N + n1*N + n2] /= sum;
                }
            }
        }

        // attn_scores x value
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                int start = int(n1 / ws) * ws;
                int end = start + ws;
                for(int d = 0; d < d_k; d++) {
                    float sum = 0.0;
                    for(int n2 = start; n2 < end; n2++) {
                        sum += attn_scores[h*N*N + n1*N + n2] * value[get(h,n2,d)];
                    }
                    result[get(h,n1,d)] = sum;
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "cpu sparse attention (window): " << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << " microseconds" << "\n\n";
    }

    void sparse_scaled_dot_product_attention_random() {
        
        float *attn_scores = (float*)malloc(N_HEAD*N*N* sizeof(float));
        float *result = (float*)malloc(N_HEAD*N*d_k * sizeof(float));

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
        std::vector<std::vector<int>> random_idx;
        random_idx.resize(N);
        for (int i = 0; i < N; i++) {
            std::vector<int> l(random_idx_tmp[i].begin(), random_idx_tmp[i].end());
            std::sort(l.begin(), l.end());
            random_idx[i] = l;
        }

        auto beg = std::chrono::steady_clock::now();

        // query x key^T
        float sqrt_d_k = sqrt(d_k);
        float score;
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                for(int n2 : random_idx[n1]) {
                    score = 0.0;
                    for(int d=0; d < d_k; d++) {
                        score += query[get(h,n1,d)] * key[get(h,n2,d)];
                    }
                    attn_scores[h*N*N + n1*N + n2] = score / sqrt_d_k;
                }
            }
        }

        // softmax on attn_scores
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                float sum = 0.0;
                for(int n2 : random_idx[n1]) {
                    sum += attn_scores[h*N*N + n1*N + n2];
                }
                for(int n2 : random_idx[n1]) {
                    attn_scores[h*N*N + n1*N + n2] /= sum;
                }
            }
        }

        // attn_scores x value
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                for(int d = 0; d < d_k; d++) {
                    float sum = 0.0;
                    for(int n2 : random_idx[n1]) {
                        sum += attn_scores[h*N*N + n1*N + n2] * value[get(h,n2,d)];
                    }
                    result[get(h,n1,d)] = sum;
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "cpu sparse attention (random): " << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << " microseconds" << "\n\n";
    }

    void scaled_dot_product_attention() {
        auto beg = std::chrono::steady_clock::now();
        float *attn_scores = (float*)malloc(N_HEAD*N*N* sizeof(float));
        float *result = (float*)malloc(N_HEAD*N*d_k * sizeof(float));

        // query x key^T
        float sqrt_d_k = sqrt(d_k);
        float score;
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                for(int n2 = 0; n2 < N; n2++) {
                    score = 0.0;
                    for(int d=0; d < d_k; d++) {
                        score += query[get(h,n1,d)] * key[get(h,n2,d)];
                    }
                    attn_scores[h*N*N + n1*N + n2] = score / sqrt_d_k;
                }
            }
        }

        // softmax on attn_scores
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                float sum = 0.0;
                for(int n2 = 0; n2 < N; n2++) {
                    sum += attn_scores[h*N*N + n1*N + n2];
                }
                for(int n2 = 0; n2 < N; n2++) {
                    attn_scores[h*N*N + n1*N + n2] /= sum;
                }
            }
        }

        // attn_scores x value
        for(int h = 0; h < N_HEAD; h++) {
            for(int n1 = 0; n1 < N; n1++) {
                for(int d = 0; d < d_k; d++) {
                    float sum = 0.0;
                    for(int n2 = 0; n2 < N; n2++) {
                        sum += attn_scores[h*N*N + n1*N + n2] * value[get(h,n2,d)];
                    }
                    result[get(h,n1,d)] = sum;
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "cpu naive attention: " << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << " microseconds" << "\n\n";
    }
};


int main(void) {
    printf("========================== CPU Naive ==========================\n");
    CPUNaiveAttention cpuNaiveAttention;
    cpuNaiveAttention.scaled_dot_product_attention();
    printf("========================== CPU Sparse (Window)==========================\n");
    cpuNaiveAttention.sparse_scaled_dot_product_attention_window();
    printf("========================== CPU Sparse (Random)==========================\n");
    cpuNaiveAttention.sparse_scaled_dot_product_attention_random();

    printf("========================== GPU Naive ==========================\n");
    gpuNaiveAttention(N, D_MODEL, N_HEAD);
    printf("========================== GPU Sparse (Window) ==========================\n");
    gpuSparseAttentionWindow(N, D_MODEL, N_HEAD);
    printf("========================== GPU Sparse (Random) ==========================\n");
    gpuSparseAttentionRandom(N, D_MODEL, N_HEAD);

	return 0;
}
