# 组合问题：N个数里面按一定规则找出k个数的集合
# 切割问题：一个字符串按一定规则有几种切割方式
# 子集问题：一个N个数的集合里有多少符合条件的子集
# 排列问题：N个数按一定规则全排列，有几种排列方式
# 棋盘问题：N皇后，解数独等等

'''
def backtracking(位置) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择当位置 in 本层集合中元素) {
        处理节点;
        backtracking(当位置 + 1); // 递归
        回溯，撤销处理结果
    }
}
'''



# 77. 组合总和
class Solution:
    def __init__(self):
        self.result = []

    def combine(self, n: int, k: int):

        self.backtracking(n, k, 1, [])
        return self.result

    def backtracking(self, n, k, startIndex, path):
        if len(path) == k:  # 满足条件
            self.result.append(path[:])
            return
        
        # 已经选择的元素个数：path.size();
        # 还需要的元素个数为: k - path.size();
        # 在集合n中至多要从该起始位置 : n - (k - path.size()) + 1，开始遍历
        for i in range(startIndex, n - (k - len(path)) + 2):  # 优化的地方
            path.append(i)  # 处理节点

            # 当前位置+1
            self.backtracking(n, k, i + 1, path)
            path.pop()  # 回溯，撤销处理的节点
    
# 93. 复原IP地址
class Solution:
    def __init__(self):
        self.res = []

    def restoreIpAddresses(self, s: str):
        
        self.backtracking(start_index=0, k=0, s=s, path=[])
        return self.res


    def judge(self, s):
        if len(s) > 1 and s[0] == '0':
            return False
        if 0 <= int(s) <= 255:
            return True
        
        return False

    def backtracking(self, start_index, k, s, path):
        # print(path)
        if k == 4:
            if start_index == len(s):
                self.res.append(".".join(path))
            return
        
        for index in range(start_index, len(s)):
            if index - start_index == 3:
                return
            if self.judge(s[start_index:index+1]):
                path.append(s[start_index:index+1])
                self.backtracking(index+1, k+1, s, path)
                path.pop()

# 78. 子集
class Solution:
    def __init__(self):
        self.res = []

    def subsets(self, nums):
        
        self.backtracking(start_index=0, nums=nums, path=[])
        return self.res

    def backtracking(self, start_index, nums, path):
        self.res.append(path[:])
        
        for index in range(start_index, len(nums)):
            path.append(nums[index])
            self.backtracking(index+1, nums, path)
            path.pop()

# 491. 递增子序列
class Solution:
    def __init__(self):
        self.res = []

    def findSubsequences(self, nums): 

        self.backtracking(startindex=0, nums=nums, path=[])
        return self.res

    def backtracking(self, startindex, nums, path):
        if len(path) >= 2:
            self.res.append(path[:])

        # 用集合去重，记录已经使用过的元素（树层去重）
        s = set()
        for index in range(startindex, len(nums)):
            if nums[index] not in s and (len(path) == 0 or nums[index] >= path[-1]):
                path.append(nums[index])
                s.add(nums[index])
                self.backtracking(index+1, nums, path)
                path.pop()

# 46. 全排列
class Solution:
    def __init__(self):
        self.res = []

    def permute(self, nums):
        
        self.backtracking(nums, [], [False] * len(nums))
        return self.res 

    def backtracking(self, nums, path, used):
        if len(path) == len(nums):
            self.res.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            self.backtracking(nums, path, used)
            path.pop()
            used[i] = False