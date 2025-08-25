import torch
import torch.quantization
from model import NNUE

# 1. 加载原始模型
device = "cpu"
model_path = "model.pth"
model = NNUE()  # 替换为你的模型类
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()  # 重要！量化前必须设为eval模式
model = torch.quantization.quantize_dynamic(
    model,  # 原始模型
    {torch.nn.Linear},  # 要量化的层类型
    dtype=torch.qint8  # 量化精度
)
# 2. 准备量化配置
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # 移动端推荐

# 3. 插入量化/反量化节点
model_prepared = torch.quantization.prepare(model)

# 4. 校准模型（用一些样本数据）
# 这里需要你准备一些校准数据
calibration_data = []  # 你的输入样本
for data in calibration_data:
    model_prepared(data)

# 5. 转换为量化模型
model_quantized = torch.quantization.convert(model_prepared)
print("量化完成")
# 6. 保存为TorchScript
traced_script = torch.jit.trace(model_quantized, torch.randn(1, 7*9*10))  # 替换input_size
traced_script.save('model_quantized.pt')
