---
title: 常见手撕算子-transpose
date: 2025-02-27 23:40:38
tags: 
- CUDA
- transpose
categories: CUDA
---
## naive版本
```cpp
__global__ void transpose_v0(float* input, float* output, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        output[col * M + row] = input[row * N + col];
    }
}
```
## 优化版本1：shared memory
思路：
1. 先将数据从global memory拷贝到shared memory中
2. 通过shared memory进行转置
3. 通过shared memory将数据拷贝到global memory中
```cpp
template <int TILE_SIZE>
__global__ void transpose_v1(float* input, float* output, int M, int N){
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; //padding to avoid bank conflicts
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if(row < M && col < N){
        tile[threadIdx.y][threadIdx.x] = input[row * N + col];
    }
    __syncthreads();

    //线程块索引交换，线程块内线程索引不变
    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    if(row < N && col < M){ //行列大小交换
        output[row * M + col] = tile[threadIdx.x][threadIdx.y];
    }
}
```
## 优化版本2：单线程处理多元素
```cpp
//BLOCK_ROWS表示每个线程块中线程的行数，即每个线程处理 TILE_SIZE/BLOCK_ROWS 个元素
template <int TILE_SIZE = 32, int BLOCK_ROWS = 8>
__global__ void transpose_v2(float* output, const float* input, int M, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int index_in = row * N + col;

    // 每个线程的每个元素跨BLOCK_ROWS行
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        if (col < N && (row + i) < M) {
            tile[threadIdx.y + i][threadIdx.x] = input[index_in + i * N];
        }
    }

    __syncthreads();

    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int index_out = row * M + col;

    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        if (col < M && (row + i) < N) {
            output[index_out + i * M] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
// M = 1024, N = 1024时，加速1.48x
```
