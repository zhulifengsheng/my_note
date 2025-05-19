from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 226. 翻转二叉树
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

# 101. 对称二叉树
def isSymmetric(root):
    if not root:
        return True

    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        elif not t1 or not t2:
            return False

        return (t1.val == t2.val) and isMirror(t1.right, t2.left) and isMirror(t1.left, t2.right)

    return isMirror(root.left, root.right)

# 450. 删除二叉搜索树中的节点
def deleteNode(root, key):
    if not root:
        return None

    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            min_larger_node = root.right
            while min_larger_node.left:
                min_larger_node = min_larger_node.left
            
            min_larger_node.left = root.left
            return root.right
    
    return root

# 108. 将有序数组转换为二叉搜索平衡树
def sortedArrayToBST(nums):
    # 按中间节点均分两半，一定可以保证是二叉搜索平衡树
    if not nums:
        return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid + 1:])

    return root

# 538. 把二叉搜索树转换为累加树
def convertBST(root):
    count = 0
    def dfs(node):
        nonlocal count
        if not node:
            return
        dfs(node.right)
        count += node.val
        node.val = count
        dfs(node.left)
    dfs(root)
    return root

# 968. 二叉树的摄像头
class Solution:
    def __init__(self):
        self.res = 0

    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # 0: 未覆盖，1: 有摄像头，2: 被覆盖
        def dps(node):
            if not node:
                return 2
            left = dps(node.left)
            right = dps(node.right)
            if left == 0 or right == 0:
                self.res += 1
                return 1
            if left == 1 or right == 1:
                return 2
            return 0
        if dps(root) == 0:
            self.res += 1
        return self.res