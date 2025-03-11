'''注意
1. mid = left + (right - left) // 2
'''

# 704. 二分法
## 1. 左闭右闭
def f1(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
        
    return -1

## 2. 左闭右开
def f2(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid
        else:
            left = mid + 1
        
    return -1

# 35. 返回元素插入的位置，如果元素不存在，返回其应该插入的位置
def f3(nums, target):
    # 左闭右闭区间
    # right的三种情况
    # 1. right = -1                 此时插入0
    # 2. right = len(nums)-1        此时插入len(nums)
    # 3. 0 <= right < len(nums)-1   此时插入right+1
    # 循环结束时，如果没有找到元素，left == right + 1（一定）

    left, right = 0, len(nums)-1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            return mid
    
    return right + 1

# 34. 寻找元素的左右边界
def f4(nums, target):
    # 定义左右边界
    leftborder, rightborder = -1, len(nums)

    # 找到元素的左边界，第一个大于等于target的位置
    left, right = 0, len(nums)-1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            # 继续向左找
            right = mid - 1
            # 左边界为这个等于target的位置
            leftborder = mid
    
    # 左边界的值没有被更新过
    if leftborder < 0:
        return [-1, -1]

    # 找到元素的右边界，第一个大于target的位置-1
    left, right = 0, len(nums)-1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            # 继续向右找
            left = mid + 1
            # 右边界为这个等于target的位置+1
            rightborder = left
    rightborder = rightborder-1
    
    if leftborder > rightborder:
        return [-1, -1]
    
    return [leftborder, rightborder]


if __name__ == "__main__":
    nums = [1, 2, 3, 4, 7, 9]
    
    res1 = f1(nums, 0)
    print(res1)

    res2 = f2(nums, 9)
    print(res2)