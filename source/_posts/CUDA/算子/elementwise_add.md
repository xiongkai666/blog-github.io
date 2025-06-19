---
title: 常见手撕算子-elementwise
date: 2025-02-27 23:40:38
tags: 
- CUDA
categories: CUDA
---
# elementwise
- elementwise 是最简单的一类算子，其指的是对数据进行逐元素操作，例如将两个等长的数组对应元素相加（add）。另外在深度学习中，激活函数会对输入数据的每个元素求对应激活值，故激活函数也算在 elementwise 范围内。
---
## add
```cpp
// 1. 向上取整
#define CEIL(a, b) ((a + b - 1) / (b))
// 2. FLOAT4，用于向量化访存，以下两种都可以
// c写法
#define FLOAT4(value) *(float4*)(&(value))
// c++写法
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

//naive版
int block_size = 1024;
int grid_size  = CEIL(N, block_size);
elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

// kernel函数
__global__ void elementwise_add(float* a, float* b, float *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

//优化方法：使用向量化访存
//使用向量化访存进行优化，需要注意，要在 grid 上除以 4：
int block_size = 1024;
int grid_size  = CEIL(CEIL(N,4), block_size);  // 注：在grid维度除以4
elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

__global__ void elementwise_add_float4(float* a, float* b, float *c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx < N) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FLOAT4(c[idx]) = tmp_c;
    }
}
```
---
## sigmoid
公式：$y = \frac{1}{1 + e^{-x}}$
```cpp
//naive版
__global__ void sigmoid0(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

//float4版

__global__ void sigmoid1(float* x, float*y, int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N){
        float4 temp_x = FLOAT4(x[idx]);
        float4 temp_y;
        temp_y.x = 1.0f / (1.0f + expf(-temp_x.x));
        temp_y.y = 1.0f / (1.0f + expf(-temp_x.y));
        temp_y.z = 1.0f / (1.0f + expf(-temp_x.z));
        temp_y.w = 1.0f / (1.0f + expf(-temp_x.w));
        FLOAT4(y[idx]) = temp_y;
    }
}
```
## relu
公式：$relu(x) = max(0, x)$
```cpp
// naive版
__global__ void relu(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
    }

// float4
__global__ void relu_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(0.0f, tmp_x.x);
        tmp_y.y = fmaxf(0.0f, tmp_x.y);
        tmp_y.z = fmaxf(0.0f, tmp_x.z);
        tmp_y.w = fmaxf(0.0f, tmp_x.w);
        FLOAT4(y[idx]) = tmp_y;
    }
}
```
