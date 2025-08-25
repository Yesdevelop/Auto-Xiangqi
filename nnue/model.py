import torch
import torch.nn as nn
import torch.ao.quantization as quantization

class NNUE(nn.Module):
    def __init__(self, input_size=7 * 9 * 10, is_quantizing=False):
        super(NNUE, self).__init__()
        self.input_size = input_size
        self.is_quantizing = is_quantizing

        if self.is_quantizing:
            self.quant = quantization.QuantStub()
        else:
            self.quant = None

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )

        if self.is_quantizing:
            self.dequant = quantization.DeQuantStub()
        else:
            self.dequant = nn.Identity()

    def forward(self, x):
        if self.is_quantizing:
            x = self.quant(x)
            x = self.fc(x)
            x = self.dequant(x)
        else:
            x = self.fc(x)
        return x