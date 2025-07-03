# 最长回文子串

def longestPalindrome(s: str) -> str:
    if len(s) < 2:
        return s
    
    # dp[i][j] 表示 s[i:j+1]是回文子串
    dp = [[0] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = 1

    res_str = s[0]
    for length in range(2, len(s)+1):
        # 定义 左右区间
        for l in range(0, len(s)):
            # R - L = length - 1
            r = l - 1 + length

            # 如果越界则终止循环
            if r >= len(s):
                break

            if s[l] != s[r]:
                dp[l][r] = 0
            else:
                if r - l < 3:
                    dp[l][r] = 1
                else:
                    dp[l][r] = dp[l+1][r-1]
            
            if dp[l][r] and len(s[l:r+1]) > len(res_str):
                res_str = s[l:r+1]

    return res_str