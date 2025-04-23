```cpp
// 1. 向上取整
#define CEIL(a, b) ((a + b - 1) / (b))

// 2. FLOAT4，用于向量化访存，以下两种都可以
// c写法
#define FLOAT4(value) *(float4*)(&(value))

// c++写法
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
```