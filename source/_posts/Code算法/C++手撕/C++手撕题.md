---
title: C++面试手撕题
date: 2025-02-27 23:40:38
tags: 
- C++
categories: C++手撕
---
# 使用C++实现一个读写锁
# 实现一个uniqued_ptr
```cpp
#include <utility> // 用于std::move

template <typename T>
class unique_ptr {
private:
    T* ptr; // 指向管理的对象

public:
    // 构造函数，接受原始指针
    explicit unique_ptr(T* p = nullptr) : ptr(p) {}

    // 析构函数，释放管理的对象
    ~unique_ptr() {
        delete ptr;
    }

    // 禁止拷贝构造函数（独占所有权不能被拷贝）
    unique_ptr(const unique_ptr&) = delete;

    // 禁止拷贝赋值运算符
    unique_ptr& operator=(const unique_ptr&) = delete;

    // 移动构造函数
    unique_ptr(unique_ptr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr; // 转移所有权后，原指针置空
    }

    // 移动赋值运算符
    unique_ptr& operator=(unique_ptr&& other) noexcept {
        if (this != &other) { // 避免自我赋值
            delete ptr;       // 释放当前管理的对象
            ptr = other.ptr;  // 接管新对象
            other.ptr = nullptr; // 原指针置空
        }
        return *this;
    }

    // 重载*运算符，获取引用
    T& operator*() const {
        return *ptr;
    }

    // 重载->运算符，获取指针
    T* operator->() const {
        return ptr;
    }

    // 获取原始指针
    T* get() const {
        return ptr;
    }

    // 重置指针，可传入新指针
    void reset(T* p = nullptr) {
        if (ptr != p) { // 避免自我重置
            delete ptr;
            ptr = p;
        }
    }

    // 释放所有权，返回原始指针
    T* release() {
        T* temp = ptr;
        ptr = nullptr; // 释放所有权后，当前指针置空
        return temp;
    }

    // 检查是否管理着对象
    explicit operator bool() const {
        return ptr != nullptr;
    }
};


// 辅助函数，用于创建unique_ptr
template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

```
# 实现一个shared_ptr
```cpp
#pragma once
#include <utility>
#include <cstddef>

template<typename T>
class SimpleSharedPtr {
private:
    struct ControlBlock {
        T* ptr;
        size_t count;
        
        explicit ControlBlock(T* p) : ptr(p), count(1) {}
        ~ControlBlock() { delete ptr; }
    };
    
    ControlBlock* controlBlock;

public:
    // 默认构造函数
    SimpleSharedPtr() : controlBlock(nullptr) {}
    
    // 构造函数
    explicit SimpleSharedPtr(T* ptr) : controlBlock(ptr ? new ControlBlock(ptr) : nullptr) {}
    
    // 拷贝构造函数
    SimpleSharedPtr(const SimpleSharedPtr& other) : controlBlock(other.controlBlock) {
        if (controlBlock) {
            ++controlBlock->count;
        }
    }
    
    // 移动构造函数
    SimpleSharedPtr(SimpleSharedPtr&& other) noexcept : controlBlock(other.controlBlock) {
        other.controlBlock = nullptr;
    }
    
    // 析构函数
    ~SimpleSharedPtr() {
        if (controlBlock && --controlBlock->count == 0) {
            delete controlBlock;
        }
    }
    
    // 拷贝赋值运算符
    SimpleSharedPtr& operator=(const SimpleSharedPtr& other) {
        if (this != &other) {
            if (controlBlock && --controlBlock->count == 0) {
                delete controlBlock;
            }
            controlBlock = other.controlBlock;
            if (controlBlock) {
                ++controlBlock->count;
            }
        }
        return *this;
    }
    
    // 移动赋值运算符
    SimpleSharedPtr& operator=(SimpleSharedPtr&& other) noexcept {
        if (this != &other) {
            if (controlBlock && --controlBlock->count == 0) {
                delete controlBlock;
            }
            controlBlock = other.controlBlock;
            other.controlBlock = nullptr;
        }
        return *this;
    }
    
    // 解引用运算符
    T& operator*() const {
        return *controlBlock->ptr;
    }
    
    // 箭头运算符
    T* operator->() const {
        return controlBlock->ptr;
    }
    
    // 获取原始指针
    T* get() const {
        return controlBlock ? controlBlock->ptr : nullptr;
    }
    
    // 获取引用计数
    size_t use_count() const {
        return controlBlock ? controlBlock->count : 0;
    }
    
    // 重置指针
    void reset() {
        if (controlBlock && --controlBlock->count == 0) {
            delete controlBlock;
        }
        controlBlock = nullptr;
    }
    
    void reset(T* ptr) {
        if (controlBlock && --controlBlock->count == 0) {
            delete controlBlock;
        }
        controlBlock = ptr ? new ControlBlock(ptr) : nullptr;
    }
};

```
## 扩展：实现一个线程安全的shared_ptr
```cpp
#include <mutex>
#include <atomic>

template<typename T>
class ThreadSafeSharedPtr {
private:
    struct ControlBlock {
        std::atomic<size_t> count;
        T* ptr;
        std::mutex mutex;

        explicit ControlBlock(T* p) : count(1), ptr(p) {}
        ~ControlBlock() { delete ptr; }
    };

    ControlBlock* controlBlock;

    void acquire() {
        if (controlBlock) {
            std::lock_guard<std::mutex> lock(controlBlock->mutex);
            ++(controlBlock->count);
        }
    }

    void release() {
        if (controlBlock) {
            bool shouldDelete = false;
            {
                std::lock_guard<std::mutex> lock(controlBlock->mutex);
                if (--(controlBlock->count) == 0) {
                    shouldDelete = true;
                }
            }
            if (shouldDelete) {
                delete controlBlock;
            }
        }
    }

public:
    // 构造函数
    explicit ThreadSafeSharedPtr(T* ptr = nullptr) : controlBlock(nullptr) {
        if (ptr) {
            controlBlock = new ControlBlock(ptr);
        }
    }

    // 拷贝构造函数
    ThreadSafeSharedPtr(const ThreadSafeSharedPtr& other) : controlBlock(other.controlBlock) {
        acquire();
    }

    // 移动构造函数
    ThreadSafeSharedPtr(ThreadSafeSharedPtr&& other) noexcept : controlBlock(other.controlBlock) {
        other.controlBlock = nullptr;
    }

    // 析构函数
    ~ThreadSafeSharedPtr() {
        release();
    }

    // 拷贝赋值运算符
    ThreadSafeSharedPtr& operator=(const ThreadSafeSharedPtr& other) {
        if (this != &other) {
            release();
            controlBlock = other.controlBlock;
            acquire();
        }
        return *this;
    }

    // 移动赋值运算符
    ThreadSafeSharedPtr& operator=(ThreadSafeSharedPtr&& other) noexcept {
        if (this != &other) {
            release();
            controlBlock = other.controlBlock;
            other.controlBlock = nullptr;
        }
        return *this;
    }

    // 解引用运算符
    T& operator*() const {
        return *(controlBlock->ptr);
    }

    // 箭头运算符
    T* operator->() const {
        return controlBlock->ptr;
    }

    // 获取原始指针
    T* get() const {
        return controlBlock ? controlBlock->ptr : nullptr;
    }

    // 获取引用计数
    size_t use_count() const {
        if (!controlBlock) return 0;
        std::lock_guard<std::mutex> lock(controlBlock->mutex);
        return controlBlock->count;
    }

    // 重置指针
    void reset(T* ptr = nullptr) {
        release();
        if (ptr) {
            controlBlock = new ControlBlock(ptr);
        } else {
            controlBlock = nullptr;
        }
    }
};
```