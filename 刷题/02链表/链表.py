class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 203. 移除链表元素
# 虚拟头方法
def remove_elements(head, val): 
    dummy = ListNode(0)
    dummy.next = head
    current = dummy

    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next

    return dummy.next

# 206. 翻转链表
def reverse_list(head):
    prev = None
    current = head

    if not head:
        return head

    while current:
        next_node = current.next  # Store the next node
        current.next = prev       # Reverse the link
        prev = current            # Move prev to current
        current = next_node       # Move to the next node

    return prev  # New head of the reversed list

# 24. 两两交换链表中的节点
def swap_pairs(head):
    if not head:
        return head

    dump = ListNode(0)
    dump.next = head
    
    current = dump
    
    while current.next:
        if not current.next.next:
            return dump.next
                
        left_node, right_node = current.next, current.next.next
        
        left_node.next = right_node.next
        right_node.next = left_node
        current.next = right_node

        current = left_node
    
    return dump.next

# 19. 删除链表的倒数第 N 个节点
# 快慢指针的方法
def delete_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head

    slow_node, fast_node = dummy, dummy
    
    for _ in range(n+1):
        fast_node = fast_node.next
    
    while fast_node:
        slow_node = slow_node.next
        fast_node = fast_node.next
    
    slow_node.next = slow_node.next.next

    return dummy.next

# 160. 相交链表
def get_intersection_node(headA, headB):
    if not headA or not headB:
        return None

    numA, numB = 0, 0
    
    cur = headA
    while cur:         # 求链表A的长度
        cur = cur.next 
        numA += 1
    
    cur = headB
    while cur:         # 求链表A的长度
        cur = cur.next 
        numB += 1

    if numB > numA:
        for _ in range(numB - numA):
            headB = headB.next
    else:
        for _ in range(numA - numB):
            headA = headA.next

    while headA:         #  遍历curA 和 curB，遇到相同则直接返回
        if headA == headB:
            return headA
        else:
            headA = headA.next
            headB = headB.next
    
    return None

# 142. 环形链表 II
# 数学
def detect_cycle(head):
    if not head:
        return None

    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next

            return slow  # The start of the cycle
    
    return None