import torch
import torch.nn as nn
from torchinfo import summary
import base
import numpy as np
import os
import json

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

def collect_json_files(num = -1):
    print("正在收集 JSON 文件...")
    result = []
    count = 0
    PATH = 'nnue/data'
    if not os.path.exists('nnue/data'):
        PATH = 'data'
    if not os.path.exists(PATH):
        raise FileNotFoundError("找不到data目录")
    for root, _, files in os.walk(PATH):
        for file in files:
            if (num > 0 and count > num):
                break
            count += 1
            if file.endswith('.json'):
                result.append(os.path.join(root, file))
    print(f"完成，找到 {len(result)} 个 JSON 文件")
    return result

def analyze_json_files(json_files):
    print("正在解析 JSON 文件...")
    result = []
    for file in json_files:
        f = open(file, 'r')
        data = f.read()
        f.close()
        data = json.loads(data)
        result.append(data)
    print(f"完成，解析了 {len(result)} 个 JSON 文件")
    return result

if __name__ == "__main__":
    # 创建模型实例 (使用默认的 hidden_size=90)
    model = NNUE()
    # json对局数据解析
    json_files = collect_json_files(10)
    datas = analyze_json_files(json_files)
    # 输出 summary
    summary(model, input_size=(1, 7 * 90))  # 假设 batch_size=1
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for data in datas:
        for ply in data:
            fen_code = ply['fen']
            for search_step in ply['data']:
                depth = search_step['depth']
                for move in search_step['data']:
                    moveid = move['moveid']
                    vl = move['vl']
                    # 真实值
                    y_train = torch.tensor([vl], dtype=torch.float32)
                    # 清除梯度
                    optimizer.zero_grad()
                    # 获取镜像的四个矩阵
                    situation = base.Situation(fen_code)
                    x_matrixes = []
                    x_matrixes.append(situation.matrix)
                    situation.flip_leftright()
                    x_matrixes.append(situation.matrix)
                    situation.flip_updown()
                    x_matrixes.append(situation.matrix)
                    situation.flip_leftright()
                    x_matrixes.append(situation.matrix)
                    # 遍历四个矩阵
                    for x_matrix in x_matrixes:
                        # 将矩阵转换为一维张量
                        x_tensor = torch.tensor(x_matrix.copy(), dtype=torch.float32)
                        x_train = x_tensor.view(1, -1)
                        # 前向传播
                        y_pred = model(x_train)
                        # 计算损失
                        loss = criterion(y_pred, y_train)
                        # 反向传播
                        loss.backward()
                        # 更新参数
                        optimizer.step()
                        # 记录损失
                        losses.append(loss.item())
                        # 每100轮打印一次损失
                        if len(losses) % 100 == 0:
                            print(f'轮次 {len(losses)}, 损失: {loss.item():.4f}')
