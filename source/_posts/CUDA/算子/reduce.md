---
title: 常见手撕算子-reduce
date: 2025-02-27 23:40:38
tags: 
- CUDA
categories: CUDA
---
Reduce 算子是指通过对数组中的每个元素进行操作，得到一个输出值的过程。常见的操作包括求和（sum）、取最大值（max）、取最小值（min）等。在 CUDA 中，优化 Reduce 算子可以显著提高计算效率。
## 1. naive实现
```cpp
//累加
__global__ void reduce1(float* d_in, float* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(d_out, d_in[idx]);
    }
}
```
## 2. share mem + 折半规约
```cpp

__global__ void reduce2(float* d_in, float* d_out, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global memory to shared mem
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = (idx < N) ? d_in[idx] : 0.0f;
    __syncthreads();
    
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s >= 1; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s]; // s = 128

        __syncthreads();
    }

    // if matrix is large, reduce one block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
    /*
    //if matrix is small, only reduce once
    if(tid == 0){
        atomicAdd(d_out, sdata[0]);
    }
    */
}
```
## 3. warp reduce
- 展开最后一个线程束，减少同步操作，提高性能。
```cpp
__device__ void warpReduce(float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}


__global__ void reduce3(float *d_in,float *d_out,int N){
    __shared__ float sdata;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? d_in[idx] : 0.0f;
        
    __syncthreads();

    #pragma unroll
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid < 32) warpReduce(sdata, tid);

    if(tid == 0) d_out[blockIdx.x] = sdata[tid];
}
```
## 4. warp shuffle
- 通过使用shfl_down_sync指令，减少同步操作，提高性能。
```cpp

__global__ void reduce4(float* d_in, float* d_out, int N) {
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    
    float sum = (idx < N) ? d_in[idx] : 0.0f;

    //do reduction in warp
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // shared mem for the sum of per warp
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;
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

//优化写法
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA核函数：使用warp级并行进行数组归约
__global__ void reduce_v3(float* d_x, float* d_y, const int N) {
    // 共享内存用于存储每个warp的部分归约结果
    __shared__ float s_y[32];  // 仅需要32个，因为一个block最多1024个线程，最多1024/32=32个warp

    // 计算全局线程索引和warp相关信息
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;      // 当前线程属于哪个warp
    int laneId = threadIdx.x % warpSize;      // 当前线程是warp中的第几个线程

    // 1. Warp内归约：每个warp独立进行归约
    float val = (idx < N) ? d_x[idx] : 0.0f;  // 搬运数据到寄存器
    
    // 使用warp shuffle指令进行高效归约
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个warp的第一个线程将归约结果存入共享内存
    if (laneId == 0) {
        s_y[warpId] = val;
    }

    __syncthreads();  // 确保所有warp的结果都已存入共享内存

    // 2. Block内归约：使用第一个warp对所有warp的结果进行最终归约
    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;  // 每个block中的warp数量
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        
        // 再次使用warp shuffle进行归约
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // 最终结果累加到输出
        if (laneId == 0) {
            atomicAdd(d_y, val);
        }
    }
}

// 调用示例
void launch_reduction(float* d_x, float* d_y, int N) {
    const int BLOCK_SIZE = 1024;  // 假设块大小为1024
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 向上取整计算网格大小
    
    reduce_v3<<<grid_size, block_size>>>(d_x, d_y, N);
}
```
## 5. 每个线程多处理几个数据
```cpp
template <unsigned int numPerThread>
__global__ void reduce5(float* d_in, float* d_out, int N) {
    
    unsigned int idx = blockIdx.x * blockDim.x * numPerThread + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;

    #pragma unroll
    for(int i = 0; i < numPerThread; i++){
        sum += (idx < N) ? d_in[idx] : 0.0f;
        idx += blockDim.x;
    }
   
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
    if(warpId == 0) {
        #pragma unroll
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    // write result for this block to global mem
    if(tid == 0) d_out[blockIdx.x] = sum;
}
```