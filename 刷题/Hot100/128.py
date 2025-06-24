def longestConsecutive(nums):
    # 遍历+去重+字典
    num_dic = {}
    for num in nums:
        num_dic[num] = True
    
    longest_streak = 0

    for num in num_dic:
        if num - 1 not in num_dic:  # 访问字典O(1)
            # 如果当前数字是一个序列的起点
            # 计算这个序列的长度
            current_num = num
            current_streak = 1
            
            while current_num + 1 in num_dic:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak
