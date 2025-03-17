
# 27. 移除元素
def f1(nums, val):
    slow_i, fast_i = 0, 0
    while fast_i < len(nums):
        if nums[fast_i] != val:
            nums[slow_i] = nums[fast_i]
            slow_i += 1
        
        fast_i += 1

    return slow_i

# 209. 滑动窗口
def f2(target, nums):
    slow_i, fast_i = 0, 0
    now_value = 0   # 记录当前值
    min_target_nums_length = float("inf")

    while fast_i < len(nums):
        now_value += nums[fast_i]
        while now_value >= target:
            min_target_nums_length = min(min_target_nums_length, fast_i - slow_i + 1)
            now_value -= nums[slow_i]
            slow_i += 1
        
        fast_i += 1

    if min_target_nums_length == float("inf"):
        return 0
    return min_target_nums_length

# 904. 滑动窗口 + 字典的使用
from collections import defaultdict, Counter
def f3(fruits):
    slow_i, fast_i = 0, 0
    res = 0
    dic = defaultdict(int)

    for fast_i, val in enumerate(fruits):
        dic[val] += 1
        while len(dic) > 2:
            dic[fruits[slow_i]] -= 1
            if dic[fruits[slow_i]] == 0:
                del dic[fruits[slow_i]]
            slow_i += 1
        res = max(res, fast_i - slow_i + 1)
    
    return res
