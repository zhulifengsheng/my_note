# 前缀和例题：求数组中某一段区间的元素和
# 给定一个数组 nums 和两个整数 left, right，求 nums[left:right+1] 的和

def prefix_sum_example(nums, left, right):
    # 计算前缀和数组
    prefix_sum = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix_sum[i + 1] = prefix_sum[i] + nums[i]
    
    # 使用前缀和快速计算区间和
    return prefix_sum[right + 1] - prefix_sum[left]

# 示例
nums = [1, 2, 3, 4, 5]
left = 1
right = 3
result = prefix_sum_example(nums, left, right)
print(f"数组 {nums} 中从索引 {left} 到 {right} 的区间和为: {result}")
# 输出: 数组 [1, 2, 3, 4, 5] 中从索引 1 到 3 的区间和为: 9

