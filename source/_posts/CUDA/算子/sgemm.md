---
title: 常见手撕算子——sgemm（单精度矩阵乘法）
date: 2025-02-27 23:40:38
tags: 
- CUDA
- softmax
- transformer
categories: CUDA
---
## 1. cpu: 矩阵乘法


```cpp
// 二维矩阵
void matrixMultiply(const float** A, const float** B, float** C, int m, int p, int n) {
    // A is m x p, B is p x n, C is m x n
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
// 二维矩阵展开成一维
void matrixMultiply(const float* A, const float* B, float* C, int m, int p, int n) {
    // A is m x p, B is p x n, C is m x n
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int k = 0; k < p; ++k) {
                // A[i][k] -> A[i * p + k]
                // B[k][j] -> B[k * n + j]
                sum += A[i * p + k] * B[k * n + j];
            }
            // C[i][j] -> C[i * n + j]
            C[i * n + j] = sum;
        }
    }
}
```
## 2. cublas: 官方矩阵乘法库
```cpp
//cublasSgemm公式：
cublasStatus_t cublasSgemm( cublasHandle_t handle, 
                            cublasOperation_t transa, cublasOperation_t transb, 
                            int m, int n, int k, 
                            const float *alpha, 
                            const float *A, int lda, 
                            const float *B, int ldb, 
                            const float *beta, 
                            float *C, int ldc);
/*
用于计算C = alpha * op(A) * op(B) + beta * C，其中handle为cublasHandle_t类型。cublas中矩阵以列优先存储，默认使用转置操作，即C^T = (A * B)^T = B^T * A^T。默认转置使用CUBLAS_OP_N，不转置使用CUBLAS_OP_T。alpha和beta为标量，A为m x k矩阵，B为k x n矩阵，C为m x n矩阵，lda、ldb、ldc为A、B、C的行数。
*/
//示例
cublasHandle_t handle;
cublasCreate(&handle);  // Initialize cuBLAS
float alpha = 1.0f;
float beta = 0.0f;
// A (2x3), B (3x2), C (2x2)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 3, &alpha, B, 2, A, 3, &beta, C, 2);
cublasDestroy(handle);
```


