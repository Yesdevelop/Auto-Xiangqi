import copy
import numpy as np
import torch
import torch.nn as nn
from convert import *

public_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"public device is {public_device}")

class nnue(nn.Module):
    def __init__(self):
        super(nnue,self).__init__()
        self.fc1 = nn.Linear(in_features=7 * 90 + 1,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=32)
        self.fc3 = nn.Linear(in_features=32,out_features=32)
        self.fc4 = nn.Linear(in_features=32,out_features=1)
        self.pc = nn.Linear(in_features=256,out_features=2086)
        self.relu = nn.ReLU()

    def forward(self,x):
        y = self.fc1(x)
        y = self.relu(y)
        #first output
        y1 = self.fc2(y)
        y1 = self.relu(y1)
        y1 = self.fc3(y1)
        y1 = self.relu(y1)
        y1 = self.fc4(y1)
        #second output
        y2 = self.pc(y)
        return y1,y2

    def convert_board_to_x(self,board : np.ndarray,side):
        inputs = np.zeros(shape=7 * 90 + 1,dtype=np.float32)
        if side == Black:
            inputs[-1] = 1
        for x in range(9):
            for y in range(10):
                p = board[x][y]
                if p < 0:
                    inputs[(abs(p) - 1) * 90 + x * 10 + y] = -1
                elif p > 0:
                    inputs[(p - 1) * 90 + x * 10 + y] = 1
        inputs = torch.from_numpy(inputs)
        return inputs.to(public_device)

if __name__ == "__main__":
    board = copy.deepcopy(init_board)
    model = nnue().to(public_device)
    inputs = model.convert_board_to_x(board,Red)
    vl,policy = model(inputs)
    print(vl)
    print(policy)
