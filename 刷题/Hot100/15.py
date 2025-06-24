def threeSum(nums):
    nums.sort()
    res = []
    
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i+1, len(nums)-1

        while left < right:
            if nums[left] + nums[right] + nums[i] > 0:
                right -= 1
            elif nums[left] + nums[right] + nums[i] < 0:
                left += 1
            else:
                res.append([nums[i], nums[left], nums[right]])

                if nums[left] == nums[right]:
                    break
                
                while nums[left] == nums[left+1]:
                    left += 1
                left += 1
                
                
                while nums[right] == nums[right-1]:
                    right -= 1
                right -= 1

    return res