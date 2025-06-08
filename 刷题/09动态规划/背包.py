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

# 343. 整数拆分
def integerBreak(n: int) -> int:
    # 1. dp为n的乘积最大值
    dp = [1] * (n+1)
    
    dp[2] = 1

    for i in range(3, n+1):
        for j in range(1, i):
            # 2. 确定递推公式：一个是j * (i - j) 直接相乘。一个是j * dp[i - j]（拆分i-j）
            dp[i] = max(dp[i], j*dp[i - j], j*(i-j))
        
        # print(dp)

    return dp[-1]

# 494. 目标和
def findTargetSumWays(nums, target: int) -> int:
    total = sum(nums)
    target = abs(target)
    
    if (total + target) % 2 != 0 or target > total:
        return 0

    # 变成几个数可以组成求和new_target
    new_target = (target + total) // 2

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