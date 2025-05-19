'''
贪心的本质是选择每一阶段的局部最优，从而达到全局最优。
这么说有点抽象，来举一个例子：
例如，有一堆钞票，你可以拿走十张，如果想达到最大的金额，你要怎么拿？
指定每次拿最大的，最终结果就是拿走最大数额的钱。
每次拿最大的就是局部最优，最后拿走最大数额的钱就是推出全局最优。
'''

'''
1. 将问题分解为若干个子问题
2. 找出适合的贪心策略
3. 求解每一个子问题的最优解
4. 将局部最优解堆叠成全局最优解
'''

# 45. 跳跃游戏 II
def jump(nums):
    if len(nums) == 1:
        return 0
        
    cur_distance = 0  # 当前覆盖最远距离下标
    ans = 0  # 记录走的最大步数
    next_distance = 0  # 下一步覆盖最远距离下标
    
    for i in range(len(nums)):
        next_distance = max(nums[i] + i, next_distance)  # 更新下一步覆盖最远距离下标
        if i == cur_distance:  # 遇到当前覆盖最远距离下标
            ans += 1  # 需要走下一步
            cur_distance = next_distance  # 更新当前覆盖最远距离下标（相当于加油了）
            if cur_distance >= len(nums) - 1:  # 当前覆盖最远距离达到数组末尾，不用再做ans++操作，直接结束
                break
    
    return ans

# 134. 加油站
def canCompleteCircuit(gas, cost):
    n = len(gas)
    total = 0  # 总油量
    cur = 0  # 当前油量
    start = 0  # 起点
    
    for i in range(n):
        total += gas[i] - cost[i]  # 累加油量
        cur += gas[i] - cost[i]  # 累加当前油量
        if cur < 0:  # 如果当前油量小于0，说明从起点到i的路程无法完成
            start = i + 1  # 更新起点为i+1
            cur = 0  # 重置当前油量
    
    return start if total >= 0 else -1

# 406. 根据身高重建队列
def reconstructQueue(people):
    people.sort(key=lambda x: (-x[0], x[1]))  # 按照身高降序排列，身高相同则按照k升序排列
    queue = []
    
    for person in people:
        queue.insert(person[1], person)  # 在k的位置插入该人
    
    return queue

# 738. 单调递增的数字
def monotoneIncreasingDigits(N):
    s = str(n)
    n = len(s)
    digits = list(s)
    
    for i in range(n - 1, 0, -1):
        if digits[i-1] > digits[i]:
            digits[i-1] = str(int(digits[i-1]) - 1)  # 减一
            for j in range(i, n):
                digits[j] = '9'  # 后面的都变成9
            # break
    
    return int(''.join(digits))
