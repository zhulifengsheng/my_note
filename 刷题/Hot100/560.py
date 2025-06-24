from collections import defaultdict

def subarraySum(nums, k: int) -> int:
    ans = s = 0
    cnt = defaultdict(int)
    for x in nums:
        cnt[s] += 1
        s += x
        ans += cnt[s - k]
    return ans
