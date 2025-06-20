'''01背包
1. dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。
2. dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
3. 注意遍历顺序，先遍历物品，再遍历背包容量
'''

'''完全背包
1. dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。
2. dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i]] + value[i])
3. 注意遍历顺序，先遍历物品，再遍历背包容量
4. 第一行的初始化
for (int j = weight[0]; j <= bagWeight; j++)
    dp[0][j] = dp[0][j - weight[0]] + value[0]; # 可以一直装物品0
'''

# 494. 目标和
def findTargetSumWays(nums, target: int) -> int:
    total = sum(nums)
    target = abs(target)
    
    if (total + target) % 2 != 0 or target > total:
        return 0

    # 变成几个数可以组成求和new_target
    new_target = (target + total) // 2

    # 目标和为0时，只有一种方式，即不选任何数
    dp = [[1] + [0] * new_target for _ in nums]
    dp[0][nums[0]] += 1

    for i in range(1, len(nums)):
        for j in range(0, new_target+1):
            if j >= nums[i]:
                dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i]]
            else:
                dp[i][j] = dp[i-1][j]
    
    return dp[-1][-1]

# 474. 一和零（滚动数组，从后向前遍历）
def findMaxForm(strs, m: int, n: int) -> int:
    # 3维数组变2维
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组，初始化为0

    # 遍历物品
    for s in strs:
        ones = s.count('1')  # 统计字符串中1的个数
        zeros = s.count('0')  # 统计字符串中0的个数
        # 遍历背包容量且从**后向前**遍历
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)  # 状态转移方程
    
    return dp[m][n]

# 377. 组合总和 IV
## 本题学习的是 “组合”和“排列”的区别
## 组合不强调顺序，(1,5)和(5,1)是同一个组合。
## 排列强调顺序，(1,5)和(5,1)是两个不同的排列。

## 如果求组合数就是外层for循环遍历物品，内层for遍历背包。
## 如果求排列数就是外层for遍历背包，内层for循环遍历物品。
def combinationSum4(nums, target: int) -> int:
    pass
    