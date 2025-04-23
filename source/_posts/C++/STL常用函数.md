---
title: C++ STL常用函数——leetcode刷题
date: 2025-02-28 17:40:38
tags: 
- STL
- 算法
categories: C++
---
# STL常用容器

## 容器分类

1. 顺序（序列式）容器：
   - `vector`：采用线性连续空间，类似于数组；
   - `deque`：双向开口的连续线性空间，随机存取，双端队列；
   - `list`：双向循环链表；
   - `slist`：单向链表；
   - `array`：固定数组，`vector`的底层即为 `array`数组。
2. 关联式容器：
   - `set`（集合）和 `map`（映射）：都是以红黑树作为底层结构。`set`不可重复，`mutliset`可重复；`map`不可重复，`mutlimap`可重复；
   - `hash_set（unordered_set）`和 `hash_map（unordered_map）`：是基于哈希表实现的，查询时间复杂度为O(1)。
3. 容器适配器：
   - `stack`：以 `deque`为底部结构并封闭其头端开口形成的；
   - `queue`：单端队列，由 `deque`实现；
   - `pirority_queue`：优先队列，类似于堆，基于 `vector`容器实现的。

## 1. string

1. 查找和替换

```cpp
int find(const string& str, int pos = 0) const; //查找str第一次出现位置,从pos开始查找
//示例
str1.find("de") != -1; //表示在str1找到"de"
```
2. 字符串存取
```cpp
back() const; //返回最后一个字符
front() const; //返回第一个字符
//示例
str1.back(); //返回str1的最后一个字符
str1.front(); //返回str1的第一个字符
```
3. 字符串比较

```cpp
int compare(const string &s) const; //与字符串s比较
//示例
s1.compare(s2) == 0; //表示s1=s2
```

4. string插入和删除

```cpp
string& insert(int pos, const string& str); //插入字符串str
string& erase(int pos, int n = npos); //删除从Pos开始的n个字符
push_back(); //尾部插入元素
```

5. string子串

```cpp
string substr(int pos = 0, int n = npos) const; //返回由pos开始的n个字符组成的字符串
// 示例
string subStr = str.substr(1, 3); //subStr等于字符串str从下标1开始的3个字符组成的子串
```

6. string转换

```cpp
//（1）数值转string
string s = to_string(val);
//（2）string转int
int n = stoi(s);
//（3）string转float
float a = stof(s)
//（4）字符char转int
int a = atoi(c);
```
7. 字符char操作
```cpp
// isalpha：判断字符是否为字母。
// isdigit：判断字符是否为数字。
// islower：判断字符是否为小写字母。
// isupper：判断字符是否为大写字母。
// tolower：将字符转换为小写。
// toupper：将字符转换为大写。
// isxdigit：判断字符是否为十六进制数字。
```
## 2.vector容器

1. vector容量和大小

```cpp
empty(); //判断容器是否为空
capacity(); //容器的容量
size(); //返回容器中元素的个数
resize(int num); //重新指定容器的长度为num，若容器变长，则以默认值0填充新位置。
//如果容器变短，则末尾超出容器长度的元素被删除。
resize(int num, elem); //重新指定容器的长度为num，若容器变长，则以elem值填充新位置。
//如果容器变短，则末尾超出容器长度的元素被删除
```

1. vector插入和删除

```cpp
push_back(ele); //尾部插入元素ele
pop_back(); //删除最后一个元素
insert(const_iterator pos, ele); //迭代器指向位置pos插入元素ele
insert(const_iterator pos, int count, ele); //迭代器指向位置pos插入count个元素ele
erase(const_iterator pos); //删除迭代器指向的元素
erase(const_iterator start, const_iterator end); //删除迭代器从start到end之间的元素
clear(); //删除容器中所有元素
```

1. vector数据存取

```cpp
at(int idx); //返回索引idx所指的数据
operator[]; //返回索引idx所指的数据
front(); //返回容器中第一个数据元素
back(); //返回容器中最后一个数据元素
```

1. vector互换容器

```cpp
swap(vec); // 将vec与本身的元素互换
//示例
v1.swap(v2); 将v1与v2的元素互换
```

## deque容器

**双端队列**，可以对头端进行插入删除操作

1. deque 插入和删除

```cpp
//两端插入操作：
push_back(elem); //在容器尾部添加一个数据
push_front(elem); //在容器头部插入一个数据
pop_back(); //删除容器最后一个数据
pop_front(); //删除容器第一个数据
//指定位置操作：
insert(pos,elem); //在pos位置插入一个elem元素的拷贝，返回新数据的位置。
insert(pos,n,elem); //在pos位置插入n个elem数据，无返回值。
insert(pos,beg,end); //在pos位置插入[beg,end)区间的数据，无返回值。
clear(); //清空容器的所有数据
erase(beg,end); //删除[beg,end)区间的数据，返回下一个数据的位置。
erase(pos); //删除pos位置的数据，返回下一个数据的位置。
```

1. deque 数据存取

```cpp
at(int idx); //返回索引idx所指的数据
operator[]; //返回索引idx所指的数据
front(); //返回容器中第一个数据元素
back(); //返回容器中最后一个数据元素
```

1. deque 排序

```cpp
sort(iterator beg, iterator end) //对beg和end区间内元素进行排序
//示例
sort(d.begin(), d.end());//对d进行排序（递增）
```

## stack容器

1. 数据存取

```cpp

push(elem); //向栈顶添加元素
emplace(elem); //同上
//插入pair数据：push({a,b}); 等价于emplace_back(a,b);
pop(); //从栈顶移除第一个元素
top(); //返回栈顶元素
```

1. 大小操作

```cpp
empty(); //判断堆栈是否为空
size(); //返回栈的大小
```

---

## queue 容器

**单端队列**允许从一端新增元素（队尾），从另一端移除元素（队头），但是两端都可以访问

1. 数据存取

```cpp
push(elem); //往队尾添加元素
pop(); //从队头移除第一个元素
back(); //返回最后一个元素（队尾）
front(); //返回第一个元素（队头）
```

1. 大小操作

```cpp
empty(); //判断堆栈是否为空
size(); //返回栈的大小
```

## 优先队列-priority_queue

priority_queue 通常基于堆（Heap）这种数据结构来实现。

使用前要包含头文件#include \<queue\>, 和queue不同的就在于我们可以自定义其中数据的优先级, 让优先级高的排在队列前面优先出队。

定义：`priority_queue<Type, Container, Functional>`，默认是大顶堆。

```cpp
// 默认是大根堆，队头最大
priority_queue <int> q;
// 大根堆
priority_queue <int, vector<int>, less<int>> q;
// 小根堆
priority_queue <int, vector<int>, greater<int>> q;
// 自定义比较方式
   // 方法一：重载运算符()
   struct cmp{ // 或者class cmp
      bool operator()(vector<int>& a, vector<int>& b){
         return a[0] > b[0];
      }
   };
   priority_queue<vector<int>, vector<vector<int>>, cmp> q; // 定义的是小根堆

   // 方法二：定义函数
   // 分为两种，普通函数不用加static，作为类成员函数要加static
   static bool cmp(vector<int>& a, vector<int>& b){
      return a[0] > b[0];
   }
   priority_queue<vector<int>, vector<vector<int>>, decltype(&cmp)> q(cmp); // 小根堆

   // 方法三：lambda函数
   auto cmp = [](vector<int>& a, vector<int>& b)->bool{
      return a[0] > b[0];
   };
   priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> q(cmp); // 小根堆

   //方法四：lambda与decltype结合，C++20
   priority_queue<vector<int>, vector<vector<int>>,
   decltype([](vector<int>& a, vector<int>& b){
        return a[0] > b[0];
   })>q;
```

和栈stack基本操作相同:

1. `top`访问队头元素
2. `empty`队列是否为空
3. `size`返回队列内元素个数
4. `push`插入元素到队尾 (并排序)
5. `emplace`原地构造一个元素并插入队列
6. `pop`弹出队头元素
7. `swap`交换内容

---

## list容器

STL中的链表是一个**双向循环**链表

1. list插入和删除

```cpp
push_back(elem);//在容器尾部加入一个元素
//push_back()可以用emplace_back()代替，且不需要创建临时元素。
pop_back();//删除容器中最后一个元素
push_front(elem);//在容器开头插入一个元素
pop_front();//从容器开头移除第一个元素
//push_front()可以用emplace_front()代替，且不需要创建临时元素。
insert(pos,elem);//在pos位置插elem元素的拷贝，返回新数据的位置。
insert(pos,n,elem);//在pos位置插入n个elem数据，无返回值。
insert(pos,beg,end);//在pos位置插入[beg,end)区间的数据，无返回值。
clear();//移除容器的所有数据
erase(beg,end);//删除[beg,end)区间的数据，返回下一个数据的位置。
erase(pos); //删除pos迭代器所指的元素，返回下一个元素的迭代器。
remove(elem);//删除容器中所有与elem值匹配的元素。
```

2. list 数据存取

```cpp
front(); //返回第一个元素。
back(); //返回最后一个元素。
```

3. list 反转和排序

```cpp
reverse(); //反转链表
sort(); //链表排序
```

4. list 移动

```cpp
1. splice(pos, other)
// 说明：pos是迭代器，other是链表。
// 功能：
将 other 链表中的所有元素移动到当前链表的 pos 位置之前。
// 函数原型：
void splice(iterator pos, list& other);
// 示例：
list1.splice(list1.begin(), list2);//将list2中的所有元素移动到list1的起始位置。

1. splice(pos, other, other_pos)
// 功能:
//将 other 链表中的 other_pos 位置元素移动到当前链表 pos 位置之前。
// 函数原型:
void splice(iterator pos, list& other, iterator it);
// 示例:
list1.splice(list1.end(), list2, list2.begin());//将list2中的第一个元素移动到list1的末尾。
// 注意：other也可以值自身，即将本身的某个元素移动到某个位置
list1.splice(list1.end(), list2, list1.begin());//将list1中的第一个元素移动到末尾。

1. splice(pos, other, other_pos, other_last)
// 功能:
//将 other 链表中的 [other_pos, other_last)区间内的元素移动到当前链表 pos 位置之前。
// 函数原型:
void splice(iterator pos, list& other, iterator first, iterator last);
// 示例:
list1.splice(list1.end(), list2, list2.begin(), list2.begin() + 2);//将list2中的前两个元素移动到list1的末尾。
// 注意：other也可以值自身，即将本身的某个元素移动到某个位置
list1.splice(list1.end(), list1, list1.begin(), list1.begin()+2);//将list1中的前两个元素移动到末尾。
```

1. list 合并
```cpp
// 函数原型:
void merge(list& x);
//功能:
将当前链表和x链表合并。
// 示例:
list1.merge(list2);//将list2中的所有元素合并到list1中。

```
## set / multiset 容器
### set基本概念
- 简介：所有元素都会在插入时自动被排序。
- 本质：set/multiset属于关联式容器，底层结构是用二叉树实现。
- set和multiset区别：set不允许容器中有重复的元素；multiset允许容器中有重复的元素。
### 函数
1. set大小和交换
```cpp
size(); //返回容器中元素的数目
empty(); //判断容器是否为空
swap(st); //交换两个集合容器
```

1. set插入和删除

```cpp
insert(elem); //在容器中插入元素。
clear(); //清除所有元素
erase(pos); //删除pos迭代器所指的元素，返回下一个元素的迭代器。
erase(beg, end); //删除区间[beg,end)的所有元素 ，返回下一个元素的迭代器。
erase(elem); //删除容器中值为elem的元素。
```

1. set查找和统计

```cpp
find(key); //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
count(key); //统计key的元素个数
```

### set和multiset区别

1. set不可以插入重复数据，而multiset可以
2. set插入数据的同时会返回插入结果，表示插入是否成功
3. multiset不会检测数据，因此可以插入重复数据

### pair对组创建

```cpp
pair<type, type> p (value1, value2);
pair<type, type> p = make_pair(value1, value2);
```

## map / multimap容器

### map基本概念

1. 简介：

- map中所有元素都是pair
- pair中第一个元素为key（键值），起到索引作用，第二个元素为value（实值）
- 所有元素都会根据元素的键值自动排序

1. 本质：
   map/multimap属于关联式容器，底层结构是用二叉树实现。
2. 优点：
   可以根据key值快速找到value值
3. map和multimap区别：

- map不允许容器中有重复key值元素
- multimap允许容器中有重复key值元素

### 函数

1. map大小和交换

```cpp
size(); //返回容器中元素的数目
empty(); //判断容器是否为空
swap(st); //交换两个集合容器
```

1. map插入和删除

```cpp
insert(elem); //在容器中插入元素。
clear(); //清除所有元素
erase(pos); //删除pos迭代器所指的元素，返回下一个元素的迭代器。
erase(beg, end); //删除区间[beg,end)的所有元素 ，返回下一个元素的迭代器。
erase(key); //删除容器中值为key的元素。
//示例：
（1）myMap[key] = value; // 如果键值不存在，则直接插入；如果键已经存在，则会更新对应的值。
（2）myMap.insert(make_pair(key, value)); // 插入一个键值对
（3）myMap.emplace(key, value);// 插入一个键值对，最优方法!
```

1. map查找和统计

```cpp
find(key); //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
count(key); //统计key的元素个数
```

# STL关系对象

## 谓词

- 返回bool类型的仿函数称为谓词
- 如果operator()接受一个参数，那么叫做一元谓词
- 如果operator()接受两个参数，那么叫做二元谓词

## 内建函数对象

### 关系仿函数

```cpp
template<class T> bool equal_to<T> //等于
template<class T> bool not_equal_to<T> //不等于
template<class T> bool greater<T> //大于
template<class T> bool greater_equal<T> //大于等于
template<class T> bool less<T> //小于
template<class T> bool less_equal<T> //小于等于
//示例
sort(v.begin(), v.end(), greater<int>()); // 对容器 v 中的元素进行降序排序
```

# STL- 常用算法

## 概述:

算法主要是由头文件 algorithm、functional和numeric组成。

- algorithm 是所有STL头文件中最大的一个，范围涉及到比较、交换、查找、遍历操作、复制、修改等等；
- numeric 体积很小，只包括几个在序列上面进行简单数学运算的模板函数；
- functional 定义了一些模板类,用以声明函数对象。

## 遍历算法

1. for_each

```cpp
for_each(iterator beg, iterator end, _func);
// 遍历算法 遍历容器元素
// beg 开始迭代器
// end 结束迭代器
// _func 函数或者函数对象
```

## 常用查找算法

1. find

```cpp
find(iterator beg, iterator end, value);
// 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
// beg 开始迭代器
// end 结束迭代器
// value 查找的元素

//示例
auto it = find(v.begin(), v.end(), 5);
```

2. find_if

```cpp
find_if(iterator beg, iterator end, _Pred);
// 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
// beg 开始迭代器
// end 结束迭代器
// _Pred 函数或者谓词（返回bool类型的仿函数）
```

3. adjacent_find

```cpp
adjacent_find(iterator beg, iterator end);
// 查找相邻重复元素,返回相邻元素的第一个位置的迭代器
// beg 开始迭代器
// end 结束迭代器
```

4.binary_search(二分查找)

```cpp
bool binary_search(iterator beg, iterator end, value);
// 查找指定的元素，查到 返回true 否则false
// 注意: 在无序序列中不可用
// beg 开始迭代器
// end 结束迭代器
// value 查找的元素
```

5. count

```cpp
count(iterator beg, iterator end, value);
// 统计元素出现次数
// beg 开始迭代器
// end 结束迭代器
// value 统计的元素
```

6. count_if

```cpp
count_if(iterator beg, iterator end, _Pred);
// 按条件统计元素出现次数
// beg 开始迭代器
// end 结束迭代器
// _Pred 谓词
```

# max_element和min_element

```cpp
max_element(iterator start, iterator end, [compare comp]);
//max_element返回一个指向范围内最大元素的迭代器，min_element返回一个指向范围内最小元素的迭代器。返回值前取 * 获取相应的最值。
//iterator start, iterator end- 这些是指向容器中范围的迭代器位置。
//[compare comp]- 它是一个可选参数，用于定义比较规则。
```

## 常用排序算法

1. sort

```cpp
sort(iterator beg, iterator end, _Pred);
// 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
// beg 开始迭代器
// end 结束迭代器
// _Pred 谓词
```

2. random_shuffle

```cpp
random_shuffle(iterator beg, iterator end);
// 指定范围内的元素随机调整次序
// beg 开始迭代器
// end 结束迭代器
```

3. merge

```cpp
merge(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);
// 容器元素合并，并存储到另一容器中
// 两个容器必须是有序的
// beg1 容器1开始迭代器 // end1 容器1结束迭代器 // beg2 容器2开始迭代器 // end2 容器2结束迭代器 //dest 目标容器开始迭代器
```

4. reverse

```cpp
reverse(iterator beg, iterator end);
// 反转指定范围的元素
// beg 开始迭代器
// end 结束迭代器
```

## 常用拷贝和替换算法

1. copy

```cpp
copy(iterator beg, iterator end, iterator dest);
// 容器内指定范围的元素拷贝到另一容器中
// beg 开始迭代器
// end 结束迭代器
// dest 目标起始迭代器
```

2. replace

```cpp
replace(iterator beg, iterator end, oldvalue, newvalue);
// 将容器内指定范围的旧元素修改为新元素
// beg 开始迭代器
// end 结束迭代器
// oldvalue 旧元素
// newvalue 新元素
```

3. replace_if

```cpp
replace_if(iterator beg, iterator end, _pred, newvalue);
// 将区间内满足条件的元素，替换成指定元素
// beg 开始迭代器
// end 结束迭代器
// _pred 谓词
// newvalue 替换的新元素
```

4. swap

```cpp
swap(container c1, container c2);
// 互换两个容器的元素
// c1容器1
// c2容器2
```

## 常用算术生成算法

算术生成算法属于小型算法，使用时包含的头文件为 #include \<numeric\>。

1. accumulate

```cpp
accumulate(iterator beg, iterator end, value);
// 计算区间内容器元素累计总和
// beg 开始迭代器
// end 结束迭代器
// value 起始值
```

2. fill

```cpp
fill(iterator beg, iterator end, value);
// 向容器中填充元素value
// beg 开始迭代器
// end 结束迭代器
// value 填充的值
```

## 常用集合算法

1. set_intersection

```cpp
set_intersection(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);
// 求两个集合的交集
// 两个集合必须是有序序列
// beg1 容器1开始迭代器 // end1 容器1结束迭代器 // beg2 容器2开始迭代器 // end2 容器2结束迭代器 //dest 目标容器开始迭代器
```

2. set_union

```cpp
set_union(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);
// 求两个集合的并集
// 两个集合必须是有序序列
// beg1 容器1开始迭代器 // end1 容器1结束迭代器 // beg2 容器2开始迭代器 // end2 容器2结束迭代器 //dest 目标容器开始迭代器
```

3. set_difference

```cpp
set_difference(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);
// 求两个集合的差集
// 两个集合必须是有序序列
// beg1 容器1开始迭代器 // end1 容器1结束迭代器 // beg2 容器2开始迭代器 // end2 容器2结束迭代器 //dest 目标容器开始迭代器
```
