import os
import torch
from model import NNUE
from board import Situation, Red, Black

# 仅使用CPU
device = "cpu"

# 与训练参数保持一致
input_size = 7 * 9 * 10
model_path = "models/epoch_1.pth"

# 初始化模型
model = NNUE(input_size=input_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device),weights_only=True))
model.to(device)
model.eval()

def evaluate(fen):
    sit = Situation(fen)
    input_tensor = torch.tensor(sit.matrix.copy(), dtype=torch.float32).view(1, -1).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # [1, 2]
    flag = Red if 'w' in fen else Black
    score = output[0, 1 - flag].item() * 1000
    return score

# 示例
if __name__ == "__main__":
    fens = [
        ["rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b - - 0 1","炮二平五"],
        ["rnbakab1r/9/1c5cn/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR w - - 0 1","炮二平五->马8进9"],
        ["rnbakab1r/9/1c4nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR w - - 0 1","炮二平五->马8进7"],
        ["rnbakab1r/9/1c5cn/p1p1C1p1p/9/9/P1P1P1P1P/1C7/9/RNBAKABNR b - - 0 1","炮二平五->马8进9->炮五进四"],
    ]
    for fen,info in fens:
        score = evaluate(fen)
        print(f'{info} => {score:.2f}')
