from itertools import permutations

def permute(nums):
    return [list(p) for p in permutations(nums,len(nums)-1)]

# 测试代码
nums = [1, 2, 3]
print(permute(nums))
    
