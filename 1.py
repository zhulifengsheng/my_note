import torch
import torch.nn as nn
import torch.nn.functional as F

# MHA
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, groups, dropout=0.0):
        pass

    def forward(self, x):
        pass

x = torch.randn(2, 4, 512) 
gqa = GroupedQueryAttention(512, 8)

output = gqa(x)
print(output.size())