以下是上述大模型推理框架所使用的主要编程语言：

### 1. vLLM
- **主要语言**：Python 和 C++。Python 用于提供高级的 API 接口，方便用户使用和集成；C++ 则用于实现底层的高性能计算逻辑，比如核心的推理算法、内存管理等，以保证框架的性能。

### 2. TensorRT - LLM
- **主要语言**：C++ 和 Python。C++ 是实现核心推理功能和与底层硬件（如 NVIDIA GPU）交互的基础，借助 CUDA 等技术进行高效计算；Python 作为前端接口，方便用户进行模型配置、参数设置和推理流程的控制。

### 3. DeepSpeed
- **主要语言**：C++ 和 Python。C++ 用于实现核心的推理算法和内存管理，Python 则用于提供高级的 API 接口，方便用户进行模型配置、参数设置和推理流程的控制。

### 4. FasterTransformer
- **主要语言**：C++ 和 CUDA C/C++。C++ 构建了框架的整体架构和逻辑，而 CUDA C/C++ 专门用于编写在 NVIDIA GPU 上运行的并行计算代码，以实现对 Transformer 模型的高效推理。

### 5. Llama.cpp
- **主要语言**：C++。该框架使用纯 C++ 编写，没有过多依赖复杂的深度学习库，具有轻量级的特点，适合在资源受限的环境中运行，并且可以充分利用 CPU 的多核性能。

## 大模型优化技术
大模型推理优化技术-KV Cache
大模型显存优化技术-PagedAttention
大模型优化技术-FlashAttention
大模型推理优化技术-Flash-Decoding
大模型显存优化技术-ZeRO系列
大模型解码优化-Speculative Decoding及其变体
大模型推理服务请求调度优化技术-Continuous batching