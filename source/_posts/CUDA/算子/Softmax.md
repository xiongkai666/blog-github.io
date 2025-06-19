---
title: 常见手撕算子——一维数组的softmax
date: 2025-02-27 23:40:38
tags: 
- CUDA
- softmax
categories: CUDA
---
# SoftMax
- Softmax 的 CPU 和 CUDA 写法均是高频考察。面试时有可能会让任选一种写法进行书写，此时自己可以先写 CPU(C++、Python) 版本，然后再写 CUDA 版本。
- Softmax公式如下：
    $$softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
- 一般为了避免溢出，需要减去最大值，所以通常采用下面这个公式：
    $$softmax(x_i) = \frac{e^{x_i - max(x)}}{\sum_j e^{x_j - max(x)}}$$
## 1. CPU(C++、Python) 版本
```cpp
    void softmax(float* input, float* output, int N){
        float max_value = *std::max_element(input, input + N);
        float sum = 0;
        for(int i = 0; i < N; i++){
            output[i] = exp(input[i] - max_value);
            sum += output[i];
        }
        for(int i = 0; i < N; i++){
            output[i] /= sum;
        }
    }
```
```python
    def softmax(input):
        max_value = max(input)
        sum = 0
        for i in range(len(input)):
            sum += exp(input[i] - max_value)
        for i in range(len(input)):
            input[i] = exp(input[i] - max_value) / sum
```

## 2. CUDA 版本
思路：
- 核函数1：归约求最值 max_val
- 核函数2：归约求和 sum
- 核函数3：计算每个元素减去 max_val 除以 sum。
```cpp

__device__ void max_kernel(float* d_in, float* d_out, int N) {
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    
    float max_value = (idx < N) ? d_in[idx] : (-FLT_MAX);

    //do reduction in warp
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
    }

    // shared mem for the sum of per warp
    const int laneId = tid % warpSize;
    const int warpId = tid / warpSize;
    int warpNum = blockDim.x / warpSize;
    __shared__ float warpLevelMaxs[warpNum];
    if(laneId == 0) warpLevelMaxs[warpId] = max_value;
    __syncthreads();

    // move data to warp0
    
    sum = (tid < warpNum)? warpLevelMaxs[tid]:(-FLT_MAX);
    // Final reduce using first warp
    if (warpId == 0){
        #pragma unroll
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
        }
    }
    // write result for this block to global mem
    if(tid == 0) d_out[blockIdx.x] = max_value;
}


__device__ void reduce_kernel(float* d_in, float* d_out, float* max_val, int N) {
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    
    float sum = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    //do reduction in warp
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // shared mem for the sum of per warp
    const int laneId = tid % warpSize;
    const int warpId = tid / warpSize;
    int warpNum = blockDim.x / warpSize;
    __shared__ float warpLevelSums[warpNum];

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // move data to warp0
    sum = (tid < warpNum)? warpLevelSums[tid]:0;
    // Final reduce using first warp
    if (warpId == 0){
        #pragma unroll
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    // write result for this block to global mem
    if(tid == 0) d_out[blockIdx.x] = sum;
}

__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);
}

//以下是主函数调用
int block_size = 256;
int grid_size  = CEIL(N, block_size);

// first block max
max_kernel<<<grid_size, block_size>>>(input, max_val, N);
// block reduce
reduce_kernel<<<grid_size, block_size>>>(input, sum, max_val, N);
softmax_kernel<<<grid_size, block_size>>>(input, output, sum, max_val, N);

```