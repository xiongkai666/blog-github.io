---
title: Python的ACM模式输出入
date: 2025-02-28 17:40:38
tags: 
- Python
- ACM
categories: 手撕
---
# 输入

## 1. 单行输入多个整数

```python
nums = list(map(int, input().split()))
print(sum(nums))
```

## 2.多行输入(行数n未知)

```python
# 方法1
while True:
    try:
        nums = list(map(int, input().split()))
        print(sum(nums))
    except EOFError:
        break

# 方法2
import sys
for line in sys.stdin:
    nums = list(map(int, line.split()))
    print(sum(nums))
```

## 3.多行输入（行数n已知）

```python
t = int(input())
for _ in range(t):
    nums = list(map(int, input().split()))
    print(sum(nums))
```

## 4. 多个测试用例，每个测试用例包含多行数据

```python
t = int(input())
for _ in range(t):
    m = int(input())
    for _ in range(m):
        nums = list(map(int, input().split()))
        print(sum(nums))
```

## 5. 复杂结构输入（如图的边）

```python
# n表示顶点数, m表示边数
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
```

## 6.字符串输入

```python
n = int(input())
for _ in range(n):
    s = input().strip()
    print(len(s))
```

## 7. 链表（将输入的数组转为单链表）

```python
class ListNode():
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

def nums_to_linklist(nums):
    if not nums:
        return None

    head = ListNode(nums[0])
    tail = head
    for i in nums[1:]:
        node = ListNode(i)
        tail.next = node
        tail = node
    return head

if __name__ == '__main__':
    nums = ['1', '2', '3', '4', '5']
    head = nums_to_linklist(nums)
    while head:
        print(head.val, end="->")
        head = head.next
    print(None)
```

## 8.二叉树（将输入的数组转为二叉树）

```python
import collections
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def array_to_binary_tree(arr:list)-> Optional[TreeNode]:
    """
    将数组转换为二叉树（层次遍历方式）
    :param arr: 输入数组，None 表示空节点，例如 [1, 2, 3, None, 4]
    :return: 二叉树根节点
    """
    if not arr:  # 空数组直接返回
        return None
  
    root = TreeNode(arr[0])  # 创建根节点
    queue = collections.deque([root])    # 初始化队列
    index = 1                # 数组指针，从第二个元素开始
  
    while queue and index < len(arr):
        node = queue.popleft()  # 取出队列最左侧节点
      
        # 处理左子节点
        if index < len(arr) and arr[index] is not None:
            node.left = TreeNode(arr[index])
            queue.append(node.left)
        index += 1  # 无论是否空节点，指针都要移动
      
        # 处理右子节点
        if index < len(arr) and arr[index] is not None:
            node.right = TreeNode(arr[index])
            queue.append(node.right)
        index += 1
  
    return root

# 层次遍历输出
def level_order(root:Optional[TreeNode]) -> Optional[TreeNode]:
    result = []
    q = collections.deque([root])
    while q:
        node = q.popleft()
        if node:
            result.append(node.val)
            q.append(node.left)
            q.append(node.right)
        else:
            result.append(None)
    # 去除末尾多余的 None
    while result and result[-1] is None:
        result.pop()
    return result

if __name__ == "__main__":
    # 输入数组（None 表示空节点）
    test_array = [1, 2, 3, 4, 5, None, 6]
  
    # 构建二叉树
    root = array_to_binary_tree(test_array)
    
    print("输入数组:", test_array)
    print("层次遍历:", level_order(root))
```

# 输出

```python
# 将结果缓存后一次性输出:
n = int(input())
res = []
for _ in range(n):
    s = input().strip()
    res.append(s)
print('\n'.join(res))
```
