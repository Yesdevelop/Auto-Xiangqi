import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import *
from convert import *

public_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    model = nnue().to(public_device)
    cross_loss = nn.CrossEntropyLoss().to(public_device)
    mse_loss = nn.MSELoss().to(public_device)
    opt = torch.optim.RAdam(model.parameters(), lr=1e-4)  # Lower the learning rate
    filepaths = get_filepaths(r"D:\dump_3", "txt")

    # Shuffle and split filepaths into training and testing sets
    np.random.shuffle(filepaths)
    split_idx = int(0.9 * len(filepaths))
    train_filepaths = filepaths[:split_idx]
    test_filepaths = filepaths[split_idx:]

    for epoch in range(10000):
        model.train()
        train_loss = 0.0
        correct_policies = 0
        total_policies = 0
        train_vl_rmse = 0.0
        total_samples = 0  # Track total number of samples processed

        # Training loop with tqdm progress bar
        train_progress = tqdm(total=len(train_filepaths), desc=f"Training Epoch {epoch+1}", unit="file")
        for path in train_filepaths:
            inputs, input_sides, vl_labels, move_ids = get_data(path)
            direct_inputs = model.convert_boards_to_xs(inputs, input_sides)
            policy_labels = torch.from_numpy(move_ids).long().to(public_device)
            vl_labels = torch.from_numpy(vl_labels).float().to(public_device).view(-1)  # Ensure vl_labels is 1D

            # Normalize inputs if needed
            direct_inputs = (direct_inputs - direct_inputs.mean()) / direct_inputs.std()

            # Process vl_labels
            vl_labels = vl_labels / 300
            vl_labels = torch.clamp(vl_labels, min=-1.0, max=1.0)

            vls, policies = model(direct_inputs)
            vls = vls.view(-1)  # Ensure vls is 1D
            policies_loss = cross_loss(policies, policy_labels)
            vl_loss = mse_loss(vls, vl_labels)
            loss = vl_loss + policies_loss * 0.1
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(inputs)  # Accumulate loss for each sample
            correct_policies += (policies.argmax(dim=1) == policy_labels).sum().item()
            total_policies += policy_labels.size(0)
            train_vl_rmse += torch.sqrt(vl_loss).item() * len(inputs)
            total_samples += len(inputs)

            # Update tqdm description with current metrics
            train_progress.set_postfix({
                "Loss": f"{train_loss / total_samples:.4f}",
                "Policy Acc": f"{correct_policies / total_policies:.4f}",
                "VL RMSE": f"{train_vl_rmse / total_samples:.4f}"
            })
            train_progress.update(1)

        avg_train_loss = train_loss / total_samples
        train_policy_accuracy = correct_policies / total_policies
        avg_train_vl_rmse = train_vl_rmse / total_samples

        # Evaluation loop with tqdm progress bar
        model.eval()
        test_loss = 0.0
        correct_policies = 0
        total_policies = 0
        test_vl_rmse = 0.0
        total_samples = 0  # Reset total_samples for testing
        test_progress = tqdm(total=len(test_filepaths), desc=f"Testing Epoch {epoch+1}", unit="file")
        with torch.no_grad():
            for path in test_filepaths:
                inputs, input_sides, vl_labels, move_ids = get_data(path)
                direct_inputs = model.convert_boards_to_xs(inputs, input_sides)
                policy_labels = torch.from_numpy(move_ids).long().to(public_device)
                vl_labels = torch.from_numpy(vl_labels).float().to(public_device).view(-1)  # Ensure vl_labels is 1D

                # Normalize inputs if needed
                direct_inputs = (direct_inputs - direct_inputs.mean()) / direct_inputs.std()

                # Process vl_labels
                vl_labels = vl_labels / 300
                vl_labels = torch.clamp(vl_labels, min=-1.0, max=1.0)

                vls, policies = model(direct_inputs)
                vls = vls.view(-1)  # Ensure vls is 1D
                policies_loss = cross_loss(policies, policy_labels)
                vl_loss = mse_loss(vls, vl_labels)
                loss = vl_loss + policies_loss * 0.1
                test_loss += loss.item() * len(inputs)  # Accumulate loss for each sample
                correct_policies += (policies.argmax(dim=1) == policy_labels).sum().item()
                total_policies += policy_labels.size(0)
                test_vl_rmse += torch.sqrt(vl_loss).item() * len(inputs)
                total_samples += len(inputs)

                # Update tqdm description with current metrics
                test_progress.set_postfix({
                    "Loss": f"{test_loss / total_samples:.4f}",
                    "Policy Acc": f"{correct_policies / total_policies:.4f}",
                    "VL RMSE": f"{test_vl_rmse / total_samples:.4f}"
                })
                test_progress.update(1)

        avg_test_loss = test_loss / total_samples
        test_policy_accuracy = correct_policies / total_policies
        avg_test_vl_rmse = test_vl_rmse / total_samples

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Policy Accuracy: {train_policy_accuracy:.4f}, Train VL RMSE: {avg_train_vl_rmse:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Policy Accuracy: {test_policy_accuracy:.4f}, Test VL RMSE: {avg_test_vl_rmse:.4f}")

        # Save model state dict with test loss in the filename
        torch.save(model.state_dict(), f"epoch_{epoch}_test_loss_{avg_test_loss:.4f}.pkl")

if __name__ == "__main__":
    train()