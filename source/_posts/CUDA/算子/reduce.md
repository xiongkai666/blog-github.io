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