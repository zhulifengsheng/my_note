def findMin(nums) -> int:
    left, right = 0, len(nums)-1

    # 终止条件是left == right
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < nums[right]:
            right = mid
        else:
            left = mid + 1

    return nums[left]
