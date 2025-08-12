import torch
import torch.nn as nn
from torchinfo import summary
import base
import numpy as np

class NNUE(nn.Module):
    def __init__(self, hidden_size = 90):
        super(NNUE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7 * 90, hidden_size),  # 第一层：630 -> hidden_size
            nn.ReLU(),                       # ReLU 激活
            nn.Linear(hidden_size, 2)        # 第二层：hidden_size -> 2
        )
        # 保存 hidden_size 作为实例属性（可选）
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # 创建模型实例 (使用默认的 hidden_size=90)
    model = NNUE()
    # 只输出 summary
    summary(model, input_size=(1, 7 * 90))  # 假设 batch_size=1
