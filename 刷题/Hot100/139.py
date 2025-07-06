# 多重背包
# 背包的循环在第二层
def wordBreak(s: str, wordDict) -> bool:
    dp = [True] + [False] * len(s)

    for pos in range(1, len(s)+1):
        for word in wordDict:
            if pos - len(word) >= 0 and s[pos-len(word):pos] == word:
                dp[pos] = dp[pos] or dp[pos-len(word)]

    return dp[-1]