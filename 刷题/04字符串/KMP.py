# TODO 感觉面试应该不会考

# 28. 实现 strStr()

def strStr(haystack, needle):
    # 1. 维护一个子串的前缀表(next_list)，避免从头再去匹配
    i, j = 0, 1
    next_list = [0]

    while j < len(needle):
        if needle[i] == needle[j]:
            i += 1
            j += 1
            next_list.append(i)
        elif i > 0:
            i = next_list[i - 1]    # 向前回退
        else:
            next_list.append(0)
            j += 1
    
    # 2. 和主串进行匹配
    i, j = 0, 0
    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1
            if j == len(needle):
                return i - j
        elif j > 0:
            j = next_list[j - 1]
        else:
            i += 1
    
    return -1


# 459. 重复的子字符串
