class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 144. 前序遍历
def preorderTraversal(root):
    res = []

    # 1. 递归法
    if not root:
        return []

    res.append(root.val)
    res.extend(preorderTraversal(root.left))
    res.extend(preorderTraversal(root.right))

    # 2. 迭代法 —— 栈
    stack = []
    stack.append(root)
    while len(stack) != 0:
        tmp_node = stack.pop()
        res.append(tmp_node.val)

        if tmp_node.right:
            stack.append(tmp_node.right)
        if tmp_node.left:
            stack.append(tmp_node.left)

    return res

# 145. 后序遍历
def postorderTraversal(root):
    res = []

    # 1. 递归法
    if not root:
        return []

    res.extend(postorderTraversal(root.left))
    res.extend(postorderTraversal(root.right))
    res.append(root.val)

    # 2. 迭代法 —— 栈
    stack = []
    stack.append(root)
    while len(stack) != 0:
        tmp_node = stack.pop()
        res.append(tmp_node.val)

        if tmp_node.left:
            stack.append(tmp_node.left)
        if tmp_node.right:
            stack.append(tmp_node.right)

    ## 注意：将整个数组反转
    res[:] = res[::-1]

    return res


# 94. 中序遍历
def inorderTraversal(root):
    res = []

    # 1. 递归法
    if not root:
        return []

    res.extend(preorderTraversal(root.left))
    res.append(root.val)
    res.extend(preorderTraversal(root.right))

    # 2. 迭代法
    stack = []

    while root or stack:
        if root:
            stack.append(root)
            root = root.left
        else:
            tmp_node = stack.pop()
            res.append(tmp_node.val)
            root = tmp_node.right
    
    # 3. 迭代（空节点法）
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        if node:
            if node.right:
                stack.append(node.right)
            
            stack.append(node)
            stack.append(None)

            if node.left:
                stack.append(node.left)
        else:
            node = stack.pop()
            res.append(node.val)

    return res

# 102. 二叉树的层序遍历(简单，不用复习)
def levelOrder(root):
    if not root:
        return []
    
    from collections import deque
    queue = deque([root])

    res = []

    while queue:
        num = len(queue)
        tmp_res = []

        for i in range(num):
            node = queue.popleft()
            tmp_res.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        res.append(tmp_res)


    return res 


if __name__ == "__main__":
    pass