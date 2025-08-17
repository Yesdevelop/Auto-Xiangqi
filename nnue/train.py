# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import json
from model import NNUE
from base import Situation, Red, Black

# 自动选择设备
public_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {public_device}")

# 创建日志目录
log_dir = "runs/nnue_dualhead"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"TensorBoard 日志将保存到: {log_dir}")

def collect_json_files(num=-1):
    """收集 JSON 数据文件"""
    print("正在收集 JSON 文件...")
    PATH = 'nnue/data' if os.path.exists('nnue/data') else 'data'
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"找不到数据目录: {PATH}")

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
    """解析所有 JSON 文件，提取 (fen, value, actor_flag) 样本"""
    print("正在解析 JSON 文件...")
    samples = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for ply in data:
                fen = ply['fen']
                current_flag = Red if 'w' in fen else Black  # 红=1, 黑=0
                for step in ply.get('data', []):
                    for move in step.get('data', []):
                        vl = float(move['vl'])
                        samples.append((fen, vl, current_flag))
        except Exception as e:
            print(f"跳过文件 {file}: {e}")
            continue
    print(f"共提取 {len(samples)} 个训练样本")
    return samples

# train.py 片段：create_dataloader
def create_dataloader(samples, batch_size=32, clip_value=1000.0):
    """生成 batch 数据，每次增强都创建独立副本，并对价值标签进行裁剪归一化"""
    indices = torch.randperm(len(samples))
    for i in range(0, len(samples), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_x, batch_y, batch_flags = [], [], []

        for idx in batch_indices:
            fen, vl, _ = samples[idx]

            # --- 关键修复：裁剪并归一化 ---
            clipped_vl = max(-clip_value, min(clip_value, vl))  # 裁剪到 [-1000, 1000]
            norm_vl = clipped_vl / clip_value  # 归一化到 [-1, 1]
            # -----------------------------

            # 四种增强
            aug_situations = [
                Situation(fen),
                Situation(fen).flip_left_and_right(),
                Situation(fen).flip_up_and_down(),
                Situation(fen).flip_left_and_right().flip_up_and_down()
            ]

            for sit in aug_situations:
                x = torch.tensor(sit.matrix, dtype=torch.float32).view(-1)
                batch_x.append(x)
                batch_y.append(norm_vl)           # 已裁剪归一化
                batch_flags.append(sit.actor_flag)

        # 转为张量
        x_batch = torch.stack(batch_x).to(public_device)
        y_batch = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(public_device)
        flags_batch = torch.tensor(batch_flags, dtype=torch.long).to(public_device)

        yield x_batch, y_batch, flags_batch

if __name__ == "__main__":
    # 超参数
    hidden_size = 90
    lr = 1e-4
    batch_size = 32
    epochs = 10
    num_files = -1  # 调试用，-1 表示全部

    # 构建模型
    model = NNUE(input_size=7*9*10, hidden_size=hidden_size).to(public_device)
    optimizer = optim.RAdam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 写入模型图（使用虚拟输入）
    dummy_input = torch.zeros(1, 7 * 9 * 10).to(public_device)
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
    total_steps = (len(samples) // batch_size + 1) * epochs

    print(f"\n开始训练，总样本数: {len(samples)}, batch_size={batch_size}, 约 {total_steps} 步")

    for epoch in range(epochs):
        print(f"\n=== 第 {epoch+1}/{epochs} 轮训练 ===")
        epoch_loss = 0.0
        step = 0

        for x_batch, y_batch, flags_batch in create_dataloader(samples, batch_size=batch_size):
            optimizer.zero_grad()

            # 前向传播: 输出 (B, 2) -> [red_score, black_score]
            all_scores = model(x_batch)  # (B, 2)

            # 使用 flags 选择对应头的输出: flags_batch 是 (B,) -> 变成 (B,1)
            selected_indices = flags_batch.unsqueeze(1)  # (B, 1)
            selected_scores = torch.gather(all_scores, 1, selected_indices)  # (B, 1)

            # 计算损失
            loss = criterion(selected_scores, y_batch)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            step += 1

            # TensorBoard 记录
            writer.add_scalar('Loss/step', loss.item(), global_step)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)

            if global_step % 100 == 0:
                print(f"步骤 {global_step}/{total_steps}, 损失: {loss.item():.6f}")

        avg_loss = epoch_loss / step
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        print(f"第 {epoch+1} 轮平均损失: {avg_loss:.6f}")

        # 更新 hparams（最后一轮）
        if epoch == epochs - 1:
            writer.add_hparams(hparams, {'hparam/final_loss': avg_loss})

    # 保存模型
    model_path = "nnue_dualhead_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n 训练完成！模型已保存至: {model_path}")
    print(f" 使用以下命令查看 TensorBoard：")
    print(f"   tensorboard --logdir={log_dir}")

    writer.close()
