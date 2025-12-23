import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import random
from dataset import H5PklDataset, collate_fn
from model import NeutraPro
from earlystopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_random_seed(seed):
    """
    Set the random seed for reproducibility across all libraries.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataloaders(h5_file_path, pkl_file_path, max_length, batch_size):
    """
    Prepare train, validation, and test dataloaders.
    Args:
        h5_file_path (str): Path to the H5 file containing embeddings.
        pkl_file_path (str): Path to the PKL file containing labels.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for the dataloaders.
    Returns:
        tuple: Train, validation, and test dataloaders.
    """
    # Load dataset
    dataset = H5PklDataset(h5_file_path, pkl_file_path, max_length)

    # Split dataset into train, validation, and test sets
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = int(0.1 * dataset_size)    # 10% for validation
    test_size = dataset_size - train_size - val_size  # 10% for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The dataloader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.
    Returns:
        tuple: Training loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print("\n开始训练...")
    
    for batch_idx, (padded_data, labels, masks) in enumerate(dataloader):
        # Move data to the same device as the model
        padded_data = padded_data.to(device)
        labels = labels.to(device).float()
        masks = masks.to(device)

        # Forward pass
        outputs = model(padded_data, mask=masks).squeeze(-1)  # Shape: (batch_size,)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy for this batch
        preds = (torch.sigmoid(outputs) > 0.5).float()
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        correct += batch_correct
        total += batch_total
        batch_acc = batch_correct / batch_total
        
        # Simple progress output
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(dataloader) - 1:
            print(f"训练批次 [{batch_idx+1}/{len(dataloader)}] 损失: {loss.item():.4f} 准确率: {batch_acc:.2f}")

    train_loss = running_loss / len(dataloader)
    train_acc = correct / total

    return train_loss, train_acc


def validate_model(model, dataloader, criterion, device):
    """
    Validate the model on the validation set.
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The dataloader for the validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use for validation.
    Returns:
        tuple: Validation loss, accuracy, and classification metrics.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\n开始验证...")

    with torch.no_grad():
        for batch_idx, (padded_data, labels, masks) in enumerate(dataloader):
            # Move data to the same device as the model
            padded_data = padded_data.to(device)
            labels = labels.to(device).float()
            masks = masks.to(device)

            # Forward pass
            outputs = model(padded_data, mask=masks).squeeze(-1)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            probs = torch.sigmoid(outputs)  # Get probabilities
            preds = (probs > 0.5).float()
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total
            batch_acc = batch_correct / batch_total

            # Collect all labels, predictions, and probabilities for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Simple progress output
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(dataloader) - 1:
                print(f"验证批次 [{batch_idx+1}/{len(dataloader)}] 损失: {loss.item():.4f} 准确率: {batch_acc:.2f}")
    
    val_loss /= len(dataloader)
    val_acc = correct / total

    # Compute classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    metrics = {
        "Validation Loss": val_loss,
        "Validation Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "all_probs": all_probs
    }

    return val_loss, val_acc, metrics


def train_model(args):
    """
    Train the NeutraPro model using data from the dataset with TensorBoard monitoring.
    Args:
        args (Namespace): Parsed command-line arguments containing all configurations.
    """
    # Set random seed
    set_random_seed(args.seed)

    # Prepare dataloaders
    train_dataloader, val_dataloader, _ = prepare_dataloaders(
        args.h5_file_path, args.pkl_file_path, args.max_length, args.batch_size
    )

    # Dynamically infer feature_dim and max_num_segments from dataloader
    for padded_data, _, _ in train_dataloader:
        _, max_num_segments, max_length, feature_dim = padded_data.shape
        break  # Only need to infer from the first batch

    print(f"Inferred parameters: feature_dim={feature_dim}, max_num_segments={max_num_segments}, max_length={max_length}")

    # Initialize the model
    model = NeutraPro(
        feature_dim=feature_dim,
        max_length=max_length,
        n_heads=args.n_heads,
        d_model=args.d_model,
        kernel_sizes=args.kernel_sizes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # AdamW with L2 regularization

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Insert after writer initialization in train_model (before the epoch loop):
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.01, path=args.model_save_path, dynamic_epochs=3)

    # Training loop
    for epoch in range(args.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)

        # Validate the model
        val_loss, val_acc, metrics = validate_model(model, val_dataloader, criterion, device)

        # Save metrics to a CSV file
        metrics["Epoch"] = epoch + 1
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(args.metrics_save_path, mode="a", header=epoch == 0, index=False)

        # Insert in the epoch loop in train_model (after obtaining metrics and before logging early stopping):
        val_precision = metrics["Precision"]
        val_recall = metrics["Recall"]
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
        print(f"Epoch {epoch+1} - Val F1: {val_f1:.6f}")

        # 绘制混淆矩阵
        cm = confusion_matrix(metrics["all_labels"], metrics["all_preds"])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        writer.add_figure("Confusion Matrix", plt.gcf(), epoch)
        plt.close()

        # 绘制预测概率分布直方图
        plt.figure(figsize=(6,4))
        plt.hist(metrics["all_probs"], bins=20, color="gray")
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        writer.add_figure("Probability Distribution", plt.gcf(), epoch)
        plt.close()

        # if early_stopping(val_f1, model):
        #     print("验证 F1 连续若干 Epoch 无提升，提前停止训练。")
        #     break

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Precision/Validation", metrics["Precision"], epoch)
        writer.add_scalar("Recall/Validation", metrics["Recall"], epoch)
        writer.add_scalar("ROC-AUC/Validation", metrics["ROC-AUC"], epoch)
        writer.add_scalar("PR-AUC/Validation", metrics["PR-AUC"], epoch)

        # Log model weights to TensorBoard
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)

        # Print validation metrics
        print("\n" + "=" * 60)  # Add a separator line for better readability
        print(f"Epoch [{epoch + 1}/{args.num_epochs}]".center(60))
        print("-" * 60)
        print(f"Validation Loss       : {val_loss:.4f}")
        print(f"Validation Accuracy   : {val_acc:.4f}")
        print(f"Precision             : {metrics['Precision']:.4f}")
        print(f"Recall                : {metrics['Recall']:.4f}")
        print(f"ROC-AUC               : {metrics['ROC-AUC']:.4f}")
        print(f"PR-AUC                : {metrics['PR-AUC']:.4f}")
        print("=" * 60)

    print("\nTraining complete.")

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NeutraPro model with configurable parameters.")
    parser.add_argument("--h5_file_path", type=str, default="data/train/embeddings.h5", help="Path to the H5 file containing embeddings.")
    parser.add_argument("--pkl_file_path", type=str, default="data/train/human_balanced.pkl", help="Path to the PKL file containing labels.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs.")
    parser.add_argument("--metrics_save_path", type=str, default="result/validation_metrics.csv", help="Path to save validation metrics.")
    parser.add_argument("--model_save_path", type=str, default="result/neutrapro.pth", help="Path to save the trained model.")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization) for AdamW optimizer.")
    parser.add_argument("--n_heads", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--d_model", type=int, default=1024, help="Dimension of the attention model.")
    parser.add_argument("--kernel_sizes", type=tuple, default=(3, 5, 7), help="Kernel sizes for the CNN layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    train_model(args)
