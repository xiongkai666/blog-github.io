---
title: C++的ACM模式输出入
date: 2025-02-28 17:40:38
tags: 
- C++
- ACM
categories: 手撕
---
# C++常用的输入方法

在C++语言中，标准输入操作需要包含头文件 `<iostream>`。

## 1. cin

cin可以连续从键盘读取多个数据，直到遇到分隔符(空格、Tab键和换行符)为止。

## 2. getline()

当我们需要读取包含空格的字符串时，cin读不全整个字符串。此时，可以使用getline()函数来解决问题。

使用getline()函数时，需要包含头文件\<string\>。getline()函数会读取整行内容，包括空格字符，直到遇到换行符为止。

函数原型：`getline(istream& is, string& str, char delim)`;
（1）`is`是输入流，这里是 `cin`，表示从标准输入流中读取数据。
（2）`str`是存储读取结果的字符串。
（3）`delim`是分隔符，即当读取到这个字符时，`getline`函数停止读取。

`getline(cin，s)`；没有提供第三个参数（分隔符），以此该句的作用是从标准输入中读取一行文本，将其存储在字符串 `s`中，直到用户按下回车键为止。

```cpp
    #include<string>
    #include<iostream>
    #include<sstream>

    string line;
    getline(cin, line); // 读取一整行
    stringstream ss(line); // 创建字符串流对象
  
    string str; //以字符串形式输出
    while (ss >> str) {
        cout << str << endl;
    }
    /*
    std::string str1;
    while (std::getline(ss, str1, ' ')) {
        std::cout << token << std::endl;
    }
    */
```

## 3. 字符串流stringstream

stringstream 是 C++ 提供的一个字符串流，使用必须包含头文件 #include\<sstream\>。
从stringstream 流中的数据输入字符串到一个变量里，是以遇到空格跳到下一个字符串的这样的形式，连续读取的。

```cpp
stringstream ss;// 创建字符串流对象
1. ss.str();//用于获取ss的当前内部字符串内容
2. stringstream ss(str1);//构造函数，将字符串str1的内容输入到字符串流ss中
3. ss.str(const string& s);//用于将指定的字符串赋给已存在的ss中
4. ss >> str2; //将字符流的内容以空格为分隔符输入到str2中，即str2得到ss的空格前的第一串内容。
5. while(ss >> str2); //按空白字符分割并逐个读取字符串
6. getline(stringtream，str，'c');//表示从每次stringstream中取出字符c之前的字符串，存入str中。
```

## 4.getchar()

getchar()函数用于从缓冲区中读取一个字符，常用于判断是否遇到换行符等场景。

```cpp
//读取直到换行符
char ch;
while ((ch = getchar()) != '\n') {
    cout << ch;
}
cout << endl;
```

# 链表的定义和输出入

```cpp
#include <iostream>

using namespace std;

// 定义链表节点的结构
struct ListNode {
    int val;         // 节点的值
    ListNode* next;  // 指向下一个节点的指针
    ListNode(int x)
        : val(x), next(nullptr) {}  // 构造函数，初始化节点的值和指针
};

int main() {
    ListNode* dummyHead = new ListNode(-1);  // 链表的虚拟头指针，初始化为空
    ListNode* tail = dummyHead;              // 链表的尾指针，初始化为空
    int n;

    // 从输入中读取数据
    while (cin >> n) {
        if (getchar() == '\n') {
            // 如果输入流中遇到换行符，则停止读取
            break;
        }
        ListNode* node = new ListNode(n);  // 创建新的链表节点

        // 将新节点添加到链表末尾
        tail->next = node;
        tail = node;

    }

    return 0;
}
```

# 二叉树的定义和输出入

## 直接从输入中获取二叉树的结点值

```cpp
#include <iostream>
#include <vector>

using namespace std;
// 定义二叉树节点的结构
struct TreeNode {
    int val;
    shared_ptr<TreeNode> left;
    shared_ptr<TreeNode> right;
    // 构造函数，初始化节点的值和子节点指针
    TreeNode(int x)
        : val(x), left(nullptr), right(nullptr) {}
};

int main() {
    int n;
    cin >> n;  // 读取树的节点个数
    if (n == 0) {
        cout << endl;  // 如果节点个数为0，直接输出换行并返回
        return 0;
    }

    vector<TreeNode* > nodes(n);
    for (int i = 0; i < n; ++i) {
        int value;
        cin >> value;
        if (value == -1) {
            nodes[i] = nullptr;  // -1 表示空节点
        } else {
            nodes[i] = new TreeNode(value);
        }
        if (i > 0) {
            if (i % 2 == 1) {
                if (nodes[(i - 1) / 2]) {
                    nodes[(i - 1) / 2]->left = nodes[i];
                }
            } else {
                if (nodes[(i - 2) / 2]) {
                    nodes[(i - 2) / 2]->right = nodes[i];
                }
            }
        }
    }

    return 0;
}
```

## 数组转换成链表

```cpp
#include<iostream>
#include<vector>

using namespace std;

struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x)
        : value(x), left(nullptr), right(nullptr) {}
};
TreeNode* createTree(vector<int>& nums, int index, int n) {
    if (index >= n || nums[index] == -1)
        return nullptr;
    TreeNode* root = new TreeNode(nums[index]);
    if (index * 2 + 1 < n)
        root->left = createTree(nums, index * 2 + 1, n);
    if (index * 2 + 2 < n)
        root->right = createTree(nums, index * 2 + 2, n);
    return root;
}
int main() {
    vector<int> nums = {1, 2, 3, 4, 5, -1, -1, -1, -1, -1};
    TreeNode* root = createTree(nums, 0, nums.size());
    return 0;
}
```
