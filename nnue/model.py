import torch
import torch.nn as nn
from torchinfo import summary

class NNUE(nn.Module):
    def __init__(self, input_size = 7 * 9 * 10,hidden_size = 90):
        super(NNUE, self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),  # 第一层：630 -> hidden_size
            nn.ReLU(),                       # ReLU 激活
            nn.Linear(in_features=hidden_size, out_features=2)        # 第二层：hidden_size -> 2
        )

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    test_device = 'cpu'
    model = NNUE().to(test_device)
    summary(model, input_size=(1, model.input_size),device=test_device)