import os
import torch
from model import NNUE
from board import Situation, Red, Black

# 仅使用CPU
device = "cpu"

# 与训练参数保持一致
input_size = 7 * 9 * 10
hidden_size = 64
model_path = "./nnue/model.pth"

# 初始化模型
model = NNUE(input_size=input_size, hidden_size=hidden_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

def evaluate(fen):
    """
    输入FEN字符串，返回模型评估值（红方视角，单位与训练一致）
    """
    sit = Situation(fen)
    input_tensor = torch.tensor(sit.matrix.copy(), dtype=torch.float32).view(1, -1).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # [1, 2]
    flag = Red if 'w' in fen else Black
    score = output[0, flag].item() * 1000
    return score

# 示例
if __name__ == "__main__":
    os.system("cls")
    fen = "1Cbakr3/4a4/2R1b4/p1p1c1p1p/9/2P1P4/P7P/9/4A4/2BK1Ar2 w - - 0 1"
    fen = "1Cbak4/4a4/4R4/p1p1c1p1p/9/2P1P4/P4r2P/9/4A4/2BK1Ar2 w - - 0 1"
    fen = "1Cbak4/9/3a5/p1p1c1p1p/9/2P1P4/P4r2P/9/4A4/2BK1Ar2 w - - 0 1"
    # fen = "2bak4/9/3a5/p1p1c1p1p/9/2P1P4/P4r2P/9/4A4/2BK1Ar2 w - - 0 1"
    # fen = "2bak4/1C7/3a5/p1p1c1p1p/9/2P1P4/P4r2P/9/4A4/2BK1Ar2 w - - 0 1"
    score = -evaluate(fen) # 我的队伍还是搞反了，需要反过来获取分数
    print(f"局面评估值: {score:.2f} (红方视角)")
