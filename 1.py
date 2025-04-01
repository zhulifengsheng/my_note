import torch
import torch.nn as nn
import torch.nn.functional as F

# MHA
class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_groups

        # 定义线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.v_linear = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # k v 重塑形状，得到的是组数，而不是头数
        k = self.k_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim)

        # 将K和V扩展到与查询头数匹配
        # (batch, seq_len, num_heads(copy了 self.num_heads // self.groups 份), head_dim)
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=2)  
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=2)

        # ==========================后面就和正常的MHA一样了===============================#
        # 调整维度顺序以便矩阵乘法
        # (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        # (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        attn_weights = F.softmax(scores, dim=-1)

        # (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v) 

        # 合并多个头
        # 转置维度: (batch_size, seq_len, num_heads, head_dim)
        context = context.transpose(1, 2).contiguous()
        # 重塑形状: (batch_size, seq_len, embed_dim)
        context = context.view(batch_size, seq_len, embed_dim)
        
        output = self.output_linear(context)

        return output

x = torch.randn(2, 4, 512) 
gqa = GroupedQueryAttention(512, 8, 2)

output = gqa(x)
print(output.size())