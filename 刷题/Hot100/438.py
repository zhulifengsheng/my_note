from collections import defaultdict

def findAnagrams(s: str, p: str):
    if len(s) < len(p):
        return []

    dic = defaultdict(int)

    res = []

    def judge(dic):
        for k, v in dic.items():
            if v != 0:
                return False
        return True

    for i in p:
        dic[i] += 1
    
    for i in range(len(p)-1):
        dic[s[i]] -= 1
    
    for i in range(len(p)-1, len(s)):
        dic[s[i]] -= 1
        if judge(dic):
            res.append(i-len(p)+1)
        
        dic[s[i-len(p)+1]] += 1

    return res

