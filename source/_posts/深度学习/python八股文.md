---
title: Python基础知识点
date: 2025-02-26 17:40:38
tags: 
- 深度学习
- Python
categories: Python
---
## 1. python的多进程，多线程，没有真正意义上的多线程，为什么这么设计
Python中"多线程没有真正意义上的并行"这一现象，核心原因在于**GIL（Global Interpreter Lock，全局解释器锁）** 的存在，这是CPython解释器（Python最主流的实现）的设计决策。要理解这一点，需要从GIL的作用、设计初衷以及多线程/多进程的本质差异说起。

### 1. 什么是GIL？
GIL是CPython解释器中的一把互斥锁，它的核心作用是：**确保同一时间只有一个线程能执行Python字节码**。

无论你的CPU有多少核心，只要在CPython解释器中，多线程执行Python代码时，都会被GIL限制——同一时刻只能有一个线程运行，其他线程必须等待GIL释放。


### 2. 为什么会有GIL？
GIL的设计是历史和现实权衡的结果，主要原因有两点：

#### （1）简化Python的内存管理
Python的内存管理（如垃圾回收机制）不是线程安全的。例如：
- Python使用引用计数跟踪对象生命周期，多线程同时操作引用计数可能导致计数混乱（如重复释放内存）。
- 早期Python的垃圾回收器（如标记-清除算法）也无法直接应对多线程竞争。

如果没有GIL，需要为每个数据结构设计细粒度锁来保证线程安全，这会极大增加解释器的复杂度，且可能导致性能下降（锁的获取/释放本身有开销）。GIL作为一把"大锁"，用最简单的方式解决了多线程下的内存安全问题。

#### （2）单线程性能优先
GIL的存在避免了细粒度锁的开销，让单线程代码运行更高效。而Python的主要应用场景（如脚本、I/O处理）对多核并行的需求并不迫切。

### 3. 为什么多线程不能真正并行，而多进程可以？
- **多线程**：所有线程共享同一个Python解释器进程，因此受同一个GIL限制。即使在多核CPU上，多线程也只能交替执行（并发），无法利用多核同时执行（并行）。
  
- **多进程**：每个进程有独立的Python解释器和内存空间，各自拥有独立的GIL。因此多个进程可以在不同CPU核心上同时运行（真正并行），不受其他进程的GIL影响。


### 4. 这种设计的适用场景与局限
GIL的设计并非"缺陷"，而是针对特定场景的取舍：
- **适合I/O密集型任务**：如网络请求、文件读写等。这类任务中，线程大部分时间在等待I/O（此时会释放GIL），其他线程可以趁机执行，多线程能有效提升效率。
- **不适合CPU密集型任务**：如大规模计算。此时多线程会因GIL竞争导致效率甚至低于单线程，需用多进程（或C扩展绕过GIL）。

### 总结
Python的多线程受GIL限制，无法实现真正的多核并行，这是CPython为了简化内存管理、优化单线程性能而做出的设计选择。而多进程通过独立的解释器和GIL，实现了真正的并行。这种设计在Python诞生时符合当时的硬件环境和使用场景，虽然后来多核CPU普及带来了局限，但也通过多进程、协程等机制提供了补充方案。