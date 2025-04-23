# C++ 常用STL总结
在 LeetCode 中使用 C++ 刷题时，**STL（Standard Template Library）** 和相关函数能极大简化代码实现。以下是高频使用的 STL 组件和函数，按类别分类整理：

---

### **一、容器类（Containers）**
1. **序列容器**  
   - **`vector`**  
     - 动态数组，支持快速随机访问  
     - 常用操作：`push_back()`, `pop_back()`, `size()`, `resize()`, `emplace_back()`, `back()`, `clear()`  
     - 场景：动态规划、数组操作、临时存储数据  

   - **`string`**  
     - 字符串处理，类似 `vector<char>`  
     - 常用操作：`substr()`, `find()`, `append()`, `push_back()`, `size()`, `empty()`  
     - 场景：回文、子串、字符串匹配  

   - **`deque`**  
     - 双端队列，支持头尾高效插入删除  
     - 场景：滑动窗口最大值（单调队列优化）  

2. **容器适配器**  
   - **`queue`**  
     - 队列，FIFO，常用操作：`push()`, `pop()`, `front()`, `empty()`  
     - 场景：BFS（广度优先搜索）  

   - **`stack`**  
     - 栈，LIFO，常用操作：`push()`, `pop()`, `top()`, `empty()`  
     - 场景：括号匹配、单调栈、DFS（非递归实现）  

   - **`priority_queue`**  
     - 优先队列（默认最大堆），常用操作：`push()`, `pop()`, `top()`  
     - 自定义排序：`priority_queue<int, vector<int>, greater<>>`（最小堆）  
     - 场景：Top K 问题、合并K个有序链表  

3. **关联容器**  
   - **`unordered_set` / `unordered_map`**  
     - 哈希表实现，查询/插入平均 O(1)  
     - 常用操作：`insert()`, `find()`, `count()`, `erase()`, `contains()`  
     - 场景：快速查找元素是否存在（如两数之和）  

   - **`set` / `map`**  
     - 红黑树实现，元素有序，查询/插入 O(log n)  
     - 场景：维护有序数据（如区间合并）  

   - **`multiset` / `multimap`**  
     - 允许重复键，其他类似 `set/map`  
     - 场景：滑动窗口中位数（维护有序序列）  

---

### **二、算法（Algorithms）**
1. **排序与查找**  
   - **`sort(begin, end, cmp)`**  
     - 快速排序，默认升序，支持自定义比较函数  
     - 场景：数组排序、贪心算法预处理  

   - **`binary_search(begin, end, val)`**  
     - 二分查找，需容器已排序  
     - 场景：有序数组查找  

2. **常用工具函数**  
   - **`reverse(begin, end)`**  
     - 反转区间元素  
     - 场景：旋转数组、字符串反转  

   - **`swap(a, b)`**  
     - 交换两个变量的值  

   - **`max(a, b)` / `min(a, b)`**  
     - 返回两个值的最大/最小值  

   - **`max_element(begin, end)`**  
     - 返回区间最大值的迭代器  

   - **`accumulate(begin, end, init)`**  
     - 计算区间元素累加和  

   - **`next_permutation(begin, end)`**  
     - 生成下一个排列  
     - 场景：全排列问题  

---

### **三、其他关键工具**
1. **迭代器与范围遍历**  
   - **`begin()`, `end()`**  
     - 获取容器的迭代器  
   - **`auto` 关键字**  
     - 简化迭代器声明：`for (auto& num : nums)`  

2. **函数对象（Functors）**  
   - **`greater<>()` / `less<>()`**  
     - 用于定义排序规则，如 `sort(v.begin(), v.end(), greater<>())`  

3. **工具类**  
   - **`pair<T1, T2>`**  
     - 存储键值对，如 `pair<int, int>` 表示坐标或区间  
   - **`tuple`**  
     - 多元素组合，较少使用但灵活  

---

### **四、典型场景与对应 STL**
- **动态规划**：`vector` 存储状态  
- **BFS**：`queue` 管理节点  
- **哈希问题**：`unordered_map` 快速查询  
- **字符串处理**：`string` 的 `substr` 和 `find`  
- **Top K 问题**：`priority_queue` 维护堆  
- **区间问题**：`map` 或 `set` 维护有序区间  

## C++ contains 函数
在 C++ 中，**`contains`** 函数的功能是检查容器或字符串中是否包含某个元素或子字符串。不过，它的具体实现和可用性取决于不同的容器类型和 C++ 标准版本。以下是详细说明：

---

### 一、字符串中的 `contains` 函数（C++23 起）
**`std::string::contains`** 是 C++23 标准新增的成员函数，用于检查字符串是否包含特定子字符串或字符。  
**语法**：
```cpp
bool contains(const std::string& substr) const;  // 检查子字符串
bool contains(char ch) const;                   // 检查字符
bool contains(const char* substr) const;        // 检查 C 风格字符串
```

**示例**：
```cpp
#include <string>
#include <iostream>

int main() {
    std::string str = "Hello, world!";
    
    // 检查子字符串
    if (str.contains("world")) {
        std::cout << "包含子字符串 'world'\n";
    }
    
    // 检查字符
    if (str.contains('o')) {
        std::cout << "包含字符 'o'\n";
    }
    
    return 0;
}
```

**兼容性问题**：  
如果编译器不支持 C++23，可以用 `find` 替代：
```cpp
if (str.find("world") != std::string::npos) {
    // 存在
}
```

---

### 二、关联容器中的 `contains` 函数（C++20 起）
C++20 标准为以下关联容器新增了 `contains` 成员函数，用于检查键是否存在：
- **`std::set`**
- **`std::map`**
- **`std::unordered_set`**
- **`std::unordered_map`**

**语法**：
```cpp
bool contains(const Key& key) const;
```

**示例**：
```cpp
#include <set>
#include <map>
#include <iostream>

int main() {
    std::set<int> s = {1, 2, 3};
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}};

    // 检查 set 中是否存在元素
    if (s.contains(2)) {
        std::cout << "set 包含 2\n";
    }

    // 检查 map 中是否存在键
    if (m.contains(3)) {
        std::cout << "map 包含键 3\n";
    } else {
        std::cout << "map 不包含键 3\n";
    }

    return 0;
}
```

**输出**：
```
set 包含 2
map 不包含键 3
```

**兼容性问题**：  
在 C++20 之前，可以使用 `find()` 替代：
```cpp
if (s.find(2) != s.end()) {
    // 存在
}
```

---

### 三、其他容器中的 `contains` 功能
对于非关联容器（如 `vector`, `list` 等），标准库没有直接提供 `contains` 函数，但可以通过以下方式实现类似功能：

#### 1. 使用 `std::find` 算法
```cpp
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int target = 3;

    if (std::find(vec.begin(), vec.end(), target) != vec.end()) {
        // 存在
    }
    return 0;
}
```

#### 2. 自定义 `contains` 模板函数
```cpp
template <typename Container, typename T>
bool contains(const Container& c, const T& value) {
    return std::find(c.begin(), c.end(), value) != c.end();
}

int main() {
    std::vector<int> vec = {1, 2, 3};
    if (contains(vec, 2)) {
        std::cout << "找到 2\n";
    }
    return 0;
}
```