# 300. 最长递增子序列
def lengthOfLIS(nums):
    pass

# 674. 最长连续递增序列
def findLengthOfLCIS(nums):
    pass

# 718. 最长重复子数组
def findLength(nums1, nums2):
    # 创建一个二维数组 dp，用于存储最长公共子数组的长度
    # 以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]。 
    # （特别注意： “以下标i - 1为结尾的A” 标明一定是 以A[i-1]为结尾的字符串 ）
    dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
    # 记录最长公共子数组的长度
    result = 0

    # 遍历数组 nums1
    for i in range(1, len(nums1) + 1):
        # 遍历数组 nums2
        for j in range(1, len(nums2) + 1):
            # 如果 nums1[i-1] 和 nums2[j-1] 相等
            if nums1[i - 1] == nums2[j - 1]:
                # 在当前位置上的最长公共子数组长度为前一个位置上的长度加一
                dp[i][j] = dp[i - 1][j - 1] + 1
            # 更新最长公共子数组的长度
            if dp[i][j] > result:
                result = dp[i][j]

    # 返回最长公共子数组的长度
    return result

# 1143. 最长公共子序列
def longestCommonSubsequence(text1: str, text2: str) -> int:
    pass

# 53.  最大子数组和
def maxSubArray(nums):
    pass

# 392.  判断子序列
def isSubsequence(s: str, t: str) -> bool:
    pass

# 115. 不同的子序列
def numDistinct(s: str, t: str) -> int:
    # 创建一个二维数组 dp，用于存储不同的子序列数量
    dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
    
    # 初始化第一列，表示空字符串 t 的子序列数量为 1
    for i in range(len(s) + 1):
        dp[i][0] = 1

    # 遍历字符串 s 和 t
    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            # 如果当前字符相等，则可以选择包含或不包含当前字符
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
            else:
                # 如果不相等，则只能选择不包含当前字符
                dp[i][j] = dp[i - 1][j]

    return dp[-1][-1]

# 583.  两个字符串的删除操作
def minDistance(word1: str, word2: str) -> int:
    dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]
    
    for i in range(len(word2) + 1):
        dp[i][0] = i
    for j in range(len(word1) + 1):
        dp[0][j] = j

    for i in range(1, len(word2) + 1):
        for j in range(1, len(word1) + 1):
            if word2[i - 1] == word1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 删除一个或者删除两个
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 2) 

    return dp[-1][-1]

# 72. 编辑距离
def minDistance(word1: str, word2: str) -> int:
    dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]
    
    for i in range(len(word2) + 1):
        dp[i][0] = i
    for j in range(len(word1) + 1):
        dp[0][j] = j

    for i in range(1, len(word2) + 1):
        for j in range(1, len(word1) + 1):
            if word2[i - 1] == word1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 删除一个或者删除两个
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1) 

    return dp[-1][-1]

# 647. 回文子串
def countSubstrings(s: str) -> int:
    dp = [[False] * len(s) for _ in range(len(s))]
    result = 0
    for i in range(len(s)-1, -1, -1): #注意遍历顺序
        for j in range(i, len(s)):
            if s[i] == s[j]:
                if j - i <= 1: #情况一 和 情况二
                    result += 1
                    dp[i][j] = True
                elif dp[i+1][j-1]: #情况三
                    result += 1
                    dp[i][j] = True
    return result

# 516. 最长回文子序列
def longestPalindromeSubseq(s: str) -> int:
    # dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]。
    dp = [[0] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = 1
    
    for i in range(len(s)-1, -1, -1):
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][-1]
