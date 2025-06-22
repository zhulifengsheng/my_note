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
## 在代码上一定将dp写成一维数据，表示从0到target有几种排列方式。
def combinationSum4(nums, target: int) -> int:
    '''
    当 1≤i≤target 时，如果存在一种排列，其中的元素之和等于 i，则该排列的最后一个元素一定是数组 nums 中的一个元素。假设该排列的最后一个元素是 num，则一定有 num≤i，对于元素之和等于 i−num 的每一种排列，在最后添加 num 之后即可得到一个元素之和等于 i 的排列，因此在计算 dp[i] 时，应该计算所有的 dp[i−num] 之和。
    '''
    dp = [1] + [0] * target

    for i in range(1, target+1):
        for num in nums:
            if num <= i:
                dp[i] += dp[i-num]

    return dp[-1]

# 213. 打家劫舍 II
# 如果是一个环形的房屋排列，不能同时偷窃第一家和最后一家。# 需要分成两种情况来处理：一种是偷窃第一家但不偷窃最后一家，另一种是偷窃最后一家但不偷窃第一
# 家。然后取这两种情况的最大值。
def rob(nums) -> int:
    if len(nums) == 1:
        return nums[0]

    dp1 = [0] * (len(nums)+1)
    dp2 = [0] * (len(nums)+1)

    for i in range(len(nums)-1):
        dp1[i+2] = max(dp1[i+1], dp1[i]+nums[i])

    for i in range(1, len(nums)):
        dp2[i+1] = max(dp2[i], dp2[i-1]+nums[i])

    return max(dp1[-1], dp2[-1])

# 309. 最佳买卖股票时机含冷冻期
def maxProfit(prices) -> int:
    n = len(prices)
    if n == 0:
        return 0
    dp = [[0] * 4 for _ in range(n)]  # 创建动态规划数组
    dp[0][0] = -prices[0]  # 初始状态：第一天持有股票的最大利润为买入股票的价格

    for i in range(1, n):
        # 当前持有股票
        dp[i][0] = max(dp[i-1][0], max(dp[i-1][3], dp[i-1][1]) - prices[i])  
        # 当前不持有股票
        dp[i][1] = max(dp[i-1][1], dp[i-1][3])
        # 当前卖出股票
        dp[i][2] = dp[i-1][0] + prices[i] 
        # 处于冷冻期
        dp[i][3] = dp[i-1][2]  
    return max(dp[n-1][3], dp[n-1][1], dp[n-1][2])  # 返回最后一天不持有股票的最大利润

# 714. 买卖股票的最佳时机含手续费
def maxProfit(prices, fee) -> int:
    n = len(prices)
    dp = [[0] * 2 for _ in range(n)]
    dp[0][0] = -prices[0] # def maxProfit(prices) -> int:

    for i in range(1, n):
        # 持有股票
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
        # 不持有股票
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
    
    return max(dp[-1])