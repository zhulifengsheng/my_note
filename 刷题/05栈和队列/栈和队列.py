from collections import deque
import heapq    # 小顶堆（优先队列）

# 维护data这个小顶堆
# data = []
# heapq.heappush(data, item)    # 加入值
# heapq.heappop(data)           # 弹出最小值



# 239. 滑动窗口最大值
def maxSlidingWindow(nums, k: int):
    # 单调队列：队列中的元素是按照从大到小的顺序排列的
    queue = deque()
    res = []

    for i in range(len(nums)):
        num = nums[i]  

        while len(queue) != 0 and queue[-1] < num:
            queue.pop()

        queue.append(num)

        if i >= k-1:
            res.append(queue[0])
            # 出队的元素
            num = nums[i-k+1]
            if queue[0] == num:
                queue.popleft()

    return res

if __name__ == "__main__":
    t = maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3)
    print(t)
    exit()

    queue = deque()

    queue.append(1)
    queue.append(2)

    # print(queue)
    print(queue.popleft())
    print(queue)