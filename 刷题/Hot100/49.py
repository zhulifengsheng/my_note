from collections import defaultdict

# 49. 字母异位词分组
def groupAnagrams(strs):
    anagrams = defaultdict(list)
    for s in strs:
        ## 解题关键在这里，排序
        key = ''.join(sorted(s))
        anagrams[key].append(s)

    return list(anagrams.values())

