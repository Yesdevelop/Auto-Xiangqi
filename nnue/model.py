import torch
import torch.nn as nn
import torch.ao.quantization as quantization

class NNUE(nn.Module):
    def __init__(self, input_size=7 * 9 * 10):
        super(NNUE, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        return self.fc(x)

class qNNUE(nn.Module):
    def __init__(self, input_size=7 * 9 * 10):
        super(qNNUE, self).__init__()
        self.input_size = input_size

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.linear1 = nn.Linear(in_features=input_size, out_features=256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=32, out_features=32)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.relu3(self.linear3(x))
        x = self.linear4(x)
        x = self.dequant(x)
        return x

    def load_weights_from_nnue(self, original_model: NNUE):
        original_state_dict = original_model.state_dict()
        quant_state_dict = self.state_dict()

        quant_state_dict['linear1.weight'].copy_(original_state_dict['fc.0.weight'])
        quant_state_dict['linear1.bias'].copy_(original_state_dict['fc.0.bias'])

        quant_state_dict['linear2.weight'].copy_(original_state_dict['fc.2.weight'])
        quant_state_dict['linear2.bias'].copy_(original_state_dict['fc.2.bias'])

        quant_state_dict['linear3.weight'].copy_(original_state_dict['fc.4.weight'])
        quant_state_dict['linear3.bias'].copy_(original_state_dict['fc.4.bias'])

        quant_state_dict['linear4.weight'].copy_(original_state_dict['fc.6.weight'])
        quant_state_dict['linear4.bias'].copy_(original_state_dict['fc.6.bias'])

        self.load_state_dict(quant_state_dict)
        print("成功从NNUE模型中读取权重并加载到qNNUE模型。")