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
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/1NBAKABNR w - - 0 1" # 让单车
    score1 = -evaluate(fen) # 我的队伍还是搞反了，需要反过来获取分数
    print(f"让单车，评估分数: {score1:.2f}")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/7C1/9/RNBAKABNR w - - 0 1" # 让单炮
    score2 = -evaluate(fen)
    print(f"让单炮，评估分数: {score2:.2f}")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/R1BAKABNR w - - 0 1" # 让单马
    score3 = -evaluate(fen)
    print(f"让单马，评估分数: {score3:.2f}")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1" # 不让
    score4 = -evaluate(fen)
    print(f"不让，评估分数: {score4:.2f}")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/9/1C5C1/9/RNBAKABNR w - - 0 1" # 让全兵
    score5 = -evaluate(fen)
    print(f"让全兵，评估分数: {score5:.2f}")
    fen = "rnbakabnr/9/9/p1p1p1p1p/9/9/P1P1P1P1P/9/9/RN2K2NR w - - 0 1" # 让士相
    score6 = -evaluate(fen)
    print(f"让士相，评估分数: {score6:.2f}")
    fen = "rnbakabnr/9/9/p1p1p1p1p/9/9/9/9/9/RN5NR w - - 0 1" # 缺帅
    score7 = -evaluate(fen)
    print(f"缺帅，评估分数: {score7:.2f}")
    fen = "9/9/9/9/9/9/9/9/9/9 w - - 0 1" # 空盘
    score8 = -evaluate(fen)
    print(f"空盘，评估分数: {score8:.2f}")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1" # 红让双车，黑让单炮
    score9 = evaluate(fen) # 黑方视角
    print(f"红让双车，黑让单炮，评估分数: {score9:.2f}")
