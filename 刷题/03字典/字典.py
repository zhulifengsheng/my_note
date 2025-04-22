# 454. 四数之和

def fourSumCount(nums1, nums2, nums3, nums4) -> int:
    from collections import Counter

    countAB = Counter(a + b for a in nums1 for b in nums2)
    countCD = Counter(c + d for c in nums3 for d in nums4)

    count = 0
    for a in countAB:
        if -a in countCD:
            count += countAB[a] * countCD[-a]
    return count
