# 15. 三数之和

def threeSum(nums):
    nums.sort()  # 排序

    result = []

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:  # 去重
            continue

        left_index = i + 1
        right_index = len(nums) - 1
        while left_index < right_index:
            # print(i, left_index, right_index)
            # print([nums[i], nums[left_index], nums[right_index]])
            if nums[left_index] + nums[i] + nums[right_index] == 0:
                result.append([nums[left_index], nums[i], nums[right_index]])

                # 去left 和 right重
                if nums[left_index] == nums[right_index]:
                    break
                while nums[right_index] == nums[right_index-1]:
                    right_index -= 1
                while nums[left_index] == nums[left_index+1]:
                    left_index += 1
                left_index += 1
                right_index -= 1

            elif nums[left_index] + nums[i] + nums[right_index] < 0:
                left_index += 1
            else:
                right_index -= 1
            
    return result

# 18. 四数之和
def fourSum(nums, target):
    nums.sort()  # 排序

    result = []

    for i in range(len(nums)):
        # 第一个结点要执行，但是后面的结点要进行去重判断
        if i > 0 and nums[i] == nums[i - 1]:  # 去重
            continue

        for j in range(i + 1, len(nums)):
            # 第一个结点要执行，但是后面的结点要进行去重判断
            if j > i + 1 and nums[j] == nums[j - 1]:  # 去重
                continue

            left_index = j + 1
            right_index = len(nums) - 1
            while left_index < right_index:
                sum_ = nums[i] + nums[j] + nums[left_index] + nums[right_index]
                if sum_ == target:
                    result.append([nums[i], nums[j], nums[left_index], nums[right_index]])

                    # 去left 和 right重
                    if nums[left_index] == nums[right_index]:
                        break
                    while nums[right_index] == nums[right_index-1]:
                        right_index -= 1
                    while nums[left_index] == nums[left_index+1]:
                        left_index += 1
                    left_index += 1
                    right_index -= 1

                elif sum_ < target:
                    left_index += 1
                else:
                    right_index -= 1

    return result