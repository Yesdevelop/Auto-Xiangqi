# train.py
import torch
import torch.nn as nn
import base
import os
import json
from model import NNUE
from torch.utils.tensorboard import SummaryWriter

# 自动选择设备
public_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {public_device}")

# 创建日志目录
log_dir = "runs/nnue_training"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"TensorBoard 日志将保存到: {log_dir}")


def collect_json_files(num=-1):
    """收集 JSON 数据文件"""
    print("正在收集 JSON 文件...")
    PATH = 'nnue/data' if os.path.exists('nnue/data') else 'data'
    if not os.path.exists(PATH):
        raise FileNotFoundError("找不到 data 目录")

    files = []
    for root, _, fs in os.walk(PATH):
        for f in sorted(fs):
            if f.endswith('.json'):
                files.append(os.path.join(root, f))
                if num > 0 and len(files) >= num:
                    break
        if num > 0 and len(files) >= num:
            break

    print(f"找到 {len(files)} 个 JSON 文件")
    return files


def analyze_json_files(json_files):
    """解析所有 JSON 文件，提取 (fen, value) 样本"""
    print("正在解析 JSON 文件...")
    samples = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for ply in data:
                fen = ply['fen']
                for step in ply.get('data', []):
                    for move in step.get('data', []):
                        vl = float(move['vl'])
                        samples.append((fen, vl))
        except Exception as e:
            print(f"跳过文件 {file}: {e}")
            continue
    print(f"共提取 {len(samples)} 个训练样本")
    return samples


def create_dataloader(samples, batch_size=32):
    """生成 batch 数据，使用 base.Situation 的翻转函数进行增强"""
    indices = torch.randperm(len(samples))
    for i in range(0, len(samples), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_x, batch_y = [], []

        for idx in batch_indices:
            fen, vl = samples[idx]
            norm_vl = vl / 1000.0  # 归一化

            # 四种镜像增强：原图 + 左右 + 上下 + 组合
            transforms = [
                lambda s: s,                                # 原图
                lambda s: s.flip_leftright(),              # 左右
                lambda s: s.flip_updown(),                 # 上下
                lambda s: s.flip_leftright().flip_updown() # 组合
            ]

            for transform in transforms:
                # 每次重建 Situation，避免污染
                situation = base.Situation(fen)
                transform(situation)  # in-place 修改副本
                mat = situation.matrix.copy()

                x = torch.tensor(mat, dtype=torch.float32).view(-1)  # (630,)
                batch_x.append(x)
                batch_y.append(norm_vl)

        # 转为张量并移动到设备
        x_batch = torch.stack(batch_x).to(public_device)
        y_batch = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(public_device)
        yield x_batch, y_batch


if __name__ == "__main__":
    # 超参数
    hidden_size = 90
    lr = 1e-4
    batch_size = 32
    epochs = 10
    num_files = 10  # 调试用，设为 -1 表示全部

    # 构建模型
    model = NNUE(input_size=7 * 9 * 10,hidden_size=hidden_size).to(public_device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 写入模型图
    dummy_input = torch.zeros(1, model.input_size).to(public_device)
    writer.add_graph(model, dummy_input)

    # 记录超参数
    hparams = {
        'hidden_size': hidden_size,
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': 'RAdam',
        'loss': 'MSELoss',
        'epochs': epochs,
        'data_files': num_files,
        'device': public_device
    }
    writer.add_hparams(hparams, {'hparam/final_loss': 0})  # 占位

    # 数据加载
    json_files = collect_json_files(num=num_files)
    samples = analyze_json_files(json_files)
    if len(samples) == 0:
        raise ValueError("无有效训练样本")

    # 训练循环
    model.train()
    global_step = 0

    for epoch in range(epochs):
        print(f"\n=== 第 {epoch+1}/{epochs} 轮训练 ===")
        epoch_loss = 0.0
        step = 0

        for x_batch, y_batch in create_dataloader(samples, batch_size=batch_size):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            step += 1

            # TensorBoard 记录
            writer.add_scalar('Loss/step', loss.item(), global_step)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)

            if global_step % 100 == 0:
                print(f"步骤 {global_step}, 损失: {loss.item():.6f}")

        avg_loss = epoch_loss / step
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        print(f"第 {epoch+1} 轮平均损失: {avg_loss:.6f}")

        # 更新 hparams（最后一轮）
        if epoch == epochs - 1:
            writer.add_hparams(hparams, {'hparam/final_loss': avg_loss})

    # 保存模型
    model_path = "nnue_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n 训练完成！模型已保存至: {model_path}")
    print(f" 使用以下命令查看 TensorBoard：")
    print(f"   tensorboard --logdir={log_dir}")

    writer.close()