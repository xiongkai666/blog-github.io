---
title: 常见手撕算子——transformer的softmax_matrix
date: 2025-02-27 23:40:38
tags: 
- CUDA
- softmax
- transformer
categories: CUDA
---
## 1.cpu: 计算每行的softmax
```cpp
void softmax_row(float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        // 第row行
        float* input_tmp  = input + row * N;
        float* output_tmp = output + row * N;
        float max_val = *(std::max_element(input_tmp, input_tmp + N));  // 计算输入数组的最大值
        float sum = 0;
        for (int i = 0; i < N; i++) {
            output_tmp[i] = std::exp(input_tmp[i] - max_val);  // 每个数先减去最大值，再求exp，避免溢出
            sum += output_tmp[i];
        }
        for (int i = 0; i < N; i++) {
            output_tmp[i] /= sum;
        }
    }
}
```
## 2. gpu: 计算每行的softmax
```cpp
__global__ void softmax_row_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x % warpSize;
    // 当前行
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // 每个线程负责计算的数据个数

    // 求每一行最大值
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;  // 最大值汇总到第一个线程，第一个线程将它搬运到s_mem

    // 求每一行的和，且要减去最大值
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // sum值汇总到第一个线程，第一个线程将它搬运到s_mem

    // 计算每一行的softmax
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}
```
## 3. 使用__shfl_xor_sync

```cpp
// gpu: 计算每行的softmax, 改用 __shfl_xor_sync 后, 每个线程的
// 寄存器的 max_val 和 sum 都是最终的结果，就不用写到共享内存再读取了
__global__ void softmax_row_kernel2(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    // 当前行
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // 每个线程负责计算的数据个数

    // 求每一行最大值
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // 求每一行的和，且要减去最大值
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // 计算每一行的softmax
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
}
```