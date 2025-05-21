'''
1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
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