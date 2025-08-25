import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NNUE
from board import Situation, Red, Black
from train import NNUEDataset, collect_json_files, create_dataloader, public_device
import os
import copy

def quantize_and_save_model(model_path, data_dir, output_path):
    model = NNUE(input_size=7 * 9 * 10,is_quantizing=True).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    quantized_model = copy.deepcopy(model)

    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(quantized_model, inplace=True)

    json_files = collect_json_files(root_path=data_dir, num=50)
    calibration_dataloader = create_dataloader(json_files, batch_size=128, num_workers=1)

    with torch.no_grad():
        for i, (x_batch, _, _) in enumerate(calibration_dataloader):
            quantized_model(x_batch)
            if i >= 10:
                break

    torch.ao.quantization.convert(quantized_model, inplace=True)

    dummy_input = torch.randn(1, 7 * 9 * 10).to('cpu')
    scripted_model = torch.jit.trace(quantized_model, dummy_input)

    scripted_model.save(output_path)

    print(f"Quantized and scripted model saved to: {output_path}")

    orig_size = os.path.getsize(model_path) / 1e6
    quant_size = os.path.getsize(output_path) / 1e6
    print(f"\nOriginal model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Size reduction: {(1 - quant_size/orig_size) * 100:.2f}%")


if __name__ == "__main__":
    trained_model_path = "models/epoch_1.pth"
    data_directory = r"F:\chess_data"
    output_model_path = "models/nnue_quantized.pt"

    if not os.path.exists(trained_model_path):
        print(f"Error: The trained model file '{trained_model_path}' was not found.")
    else:
        try:
            quantize_and_save_model(
                model_path=trained_model_path,
                data_dir=data_directory,
                output_path=output_model_path
            )
        except Exception as e:
            print(f"An error occurred during quantization: {e}")