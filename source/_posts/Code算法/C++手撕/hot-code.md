---
title: 面试和笔试中经常出现的（模板）代码题
date: 2025-02-27 17:40:38
tags: 
- C++
- 算法
- leetcode
categories: 面试手撕
---
# 1. 实现LRUCache
```cpp
// 方法一：不使用STL，自己设计数据结构
#include<unordered_map>
#include<iostream>
using namespace std;
struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode()
        : key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value)
        : key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity)
        : capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }

    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = tail->prev;
                removeNode(removed);
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        } else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        head->next->prev = node;
        node->next = head->next;
        node->prev = head;
        head->next = node;
    }

    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }
};

//方法二：使用STL
/*
class LRUCache {
private:
    int capacity;
    list<pair<int,int>> cacheList;
    unordered_map<int, list<pair<int,int>>::iterator> cacheMap;
public:
    LRUCache(int capacity) {
        this->capacity = capacity;
    }
    
    int get(int key) {
        if(cacheMap.count(key)){
            cacheList.splice(cacheList.begin(), cacheList, cacheMap[key]);
            return cacheMap[key]->second;
        }else{
            return -1;
        }
    }
    
    void put(int key, int value) {
        if(cacheMap.count(key)){
            cacheMap[key]->second = value;
            cacheList.splice(cacheList.begin(), cacheList, cacheMap[key]);
        }else{
            if(cacheList.size() == capacity){
                cacheMap.erase(cacheList.back().first);
                cacheList.pop_back();
            }
            cacheList.emplace_front(key, value);
            cacheMap[key] = cacheList.begin();
        }
    }
};
*/
int main() {
    LRUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cout << cache.get(1) << endl;  // 输出 1
    cache.put(3, 3);               // 移除 key 2
    cout << cache.get(2) << endl;  // 输出 -1 (未找到)
    cache.put(4, 4);               // 移除 key 1
    cout << cache.get(1) << endl;  // 输出 -1 (未找到)
    cout << cache.get(3) << endl;  // 输出 3
    cout << cache.get(4) << endl;  // 输出 4
    return 0;
}
```
# 2. 快速排序
```cpp
#include <iostream>
#include <vector>

using namespace std;

//方法一：空穴法，课本上的方法。（推荐）
void QuickSort(vector<int>& nums, int l, int r) {
    if (l >= r)
        return;  // 此时已经完成排序，直接返回
    int pivot = nums[l], i = l, j = r;
    while (i < j) {
        while (i < j && nums[j] >= pivot) j--;
        nums[i] = nums[j];
        while (i < j && nums[i] <= pivot) i++;
        nums[j] = nums[i];
    }
    nums[i] = pivot;      // 将枢轴（pivot）元素移动到正确位置
    QuickSort(nums, l, i - 1);   // 左边子序列递归排序
    QuickSort(nums, i + 1, r);  // 右边子序列递归排序
}

// 方法二：左右指针法
void QuickSort(vector<int>& nums, int l, int r) {
    if (l >= r)
        return;  // 此时已经完成排序，直接返回
    int pivot = nums[l], i = l, j = r;
    while (i < j) {
        while (i < j && nums[j] >= pivot)
            j--;
        while (i < j && nums[i] <= pivot)
            i++;
        swap(nums[i], nums[j]);
    }
    swap(nums[l], nums[i]);     // 将枢轴（pivot）元素移动到正确位置
    QuickSort(nums, l, i - 1);  // 左边子序列递归排序
    QuickSort(nums, i + 1, r);  // 右边子序列递归排序
}

int main() {
    vector<int> nums = {2, 1, 3, 5, 8, 2, 9, 1, 4, 7, 6, 0};
    QuickSort(nums, 0, nums.size() - 1);
    for (int num : nums)
        cout << num << " ";  // 输出0 1 2 3 4 5 6 7 8 9
    return 0;
}
```
### 快速排序的扩展：
```cpp
// 快速选择算法（快速排序的变种），找出第k大的数
int quickselect(vector<int> &nums, int l, int r, int k) {
    if (l >= r)
        return nums[k];
    int partition = nums[l], i = l - 1, j = r + 1;
    while (i < j) {
        do i++; while (nums[i] < partition);
        do j--; while (nums[j] > partition);
        if (i < j)
            swap(nums[i], nums[j]);
    }
    if (k <= j)return quickselect(nums, l, j, k);
    else return quickselect(nums, j + 1, r, k);
}
int findKthLargest(vector<int> &nums, int k) {
        int n = nums.size();
        return quickselect(nums, 0, n - 1, n - k); //寻找第k大的数
    }
```