import os
import logging
from typing import List, Dict, Tuple

import numpy as np
from numpy.linalg import norm
from numpy.ma.core import filled
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    """
    EEG Dataset for training and evaluation.

    Args:
        data (np.array): Array of EEG signals.
        labels (np.array): Corresponding labels for the EEG signals.
    """
    def __init__(self, data: np.array, labels: np.array):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

class EEGModel(nn.Module):
    """
    EEG Classification Model using Convolutional Neural Network.

    Args:
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes: int):
        super(EEGModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EEGTrainer:
    """
    Trainer class for the EEG classification model.

    Args:
        model (nn.Module): EEG classification model.
        device (str): Device to use for training ('cpu' or 'cuda').
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train for.
    """
    def __init__(self, model: nn.Module, device: str, num_classes: int, learning_rate: float, batch_size: int, num_epochs: int):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader: DataLoader):
        """
        Train the EEG classification model.

        Args:
            train_loader (DataLoader): Data loader for training data.
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate the EEG classification model.

        Args:
            val_loader (DataLoader): Data loader for validation data.

        Returns:
            float: Average loss on the validation set.
            float: Classification accuracy on the validation set.
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += torch.eq(preds, labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / len(val_loader.dataset)
        logger.info(f"Validation Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        return avg_loss, accuracy

def align_tangent_spaces(source_data: np.array, target_data: np.array, source_labels: np.array, target_labels: np.array,
                       num_classes: int, device: str):
    """
    Align tangent spaces between source and target EEG data.

    Args:
        source_data (np.array): EEG data from the source domain.
        target_data (np.array): EEG data from the target domain.
        source_labels (np.array): Labels for the source EEG data.
        target_labels (np.array): Labels for the target EEG data.
        num_classes (int): Number of output classes.
        device (str): Device to use for training ('cpu' or 'cuda').

    Returns:
        np.array: Aligned source EEG data.
        np.array: Aligned target EEG data.
    """
    # Create EEG dataset and data loaders
    source_dataset = EEGDataset(source_data, source_labels)
    target_dataset = EEGDataset(target_data, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

    # Create and train EEG classification model
    model = EEGModel(num_classes)
    trainer = EEGTrainer(model, device, num_classes, learning_rate=0.001, batch_size=32, num_epochs=10)
    trainer.train(source_loader)

    # Get source and target embeddings
    source_embeddings = []
    target_embeddings = []
    model.eval()
    with torch.no_grad():
        for data, _ in source_loader:
            data = data.to(device)
            embeddings = model.fc1(model.dropout(F.relu(model.fc1(model.dropout(F.relu(model.fc1(F.relu(model.conv2(F.relu(model.conv1(data)))))))))))
            source_embeddings.append(embeddings.cpu().numpy())

        for data, _ in target_loader:
            data = data.to(device)
            embeddings = model.fc1(model.dropout(F.relu(model.fc1(model.dropout(F.relu(model.fc1(F.relu(model.conv2(F.relu(model.conv1(data)))))))))))
            target_embeddings.append(embeddings.cpu().numpy())

    source_embeddings = np.vstack(source_embeddings)
    target_embeddings = np.vstack(target_embeddings)

    # Align tangent spaces
    mean_source = np.mean(source_embeddings, axis=0)
    mean_target = np.mean(target_embeddings, axis=0)
    source_cov = np.cov(source_embeddings.T)
    target_cov = np.cov(target_embeddings.T)
    source_inv_cov = np.linalg.inv(source_cov + 1e-5 * np.eye(source_cov.shape[0]))
    target_inv_cov = np.linalg.inv(target_cov + 1e-5 * np.eye(target_cov.shape[0]))
    aligned_source_embeddings = source_embeddings @ (target_inv_cov @ source_inv_cov) @ (target_embeddings - mean_target).T @ (source_embeddings - mean_source) / norm(source_embeddings - mean_source, axis=1).reshape(-1, 1) + mean_target
    aligned_target_embeddings = target_embeddings

    # Replace original EEG data with aligned embeddings
    aligned_source_data = np.zeros_like(source_data)
    aligned_target_data = np.zeros_like(target_data)
    for i in range(source_data.shape[0]):
        aligned_source_data[i, :] = aligned_source_embeddings[i, :]
    for i in range(target_data.shape[0]):
        aligned_target_data[i, :] = aligned_target_embeddings[i, :]

    return aligned_source_data, aligned_target_data

def spatial_riemannian_feature_fusion(source_data: np.array, target_data: np.array, source_labels: np.array, target_labels: np.array,
                                     num_classes: int, device: str):
    """
    Perform spatial Riemannian feature fusion between source and target EEG data.

    Args:
        source_data (np.array): EEG data from the source domain.
        target_data (np.array): EEG data from the target domain.
        source_labels (np.array): Labels for the source EEG data.
        target_labels (np.array): Labels for the target EEG data.
        num_classes (int): Number of output classes.
        device (str): Device to use for training ('cpu' or 'cuda').

    Returns:
        np.array: Fused source and target EEG data.
    """
    # Align tangent spaces
    aligned_source_data, aligned_target_data = align_tangent_spaces(source_data, target_data, source_labels, target_labels, num_classes, device)

    # Concatenate aligned source and target data
    fused_data = np.vstack((aligned_source_data, aligned_target_data))

    return fused_data

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load and preprocess data
    data_path = "eeg_data.csv"
    df = pd.read_csv(data_path)
    eeg_data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # Split data into source and target domains
    source_data = eeg_data[:len(eeg_data)//2]
    source_labels = labels[:len(labels)//2]
    target_data = eeg_data[len(eeg_data)//2:]
    target_labels = labels[len(labels)//2:]

    # Perform cross-subject and cross-montage EEG transfer learning
    num_classes = len(np.unique(labels))
    fused_data = spatial_riemannian_feature_fusion(source_data, target_data, source_labels, target_labels, num_classes, device)

    # Save fused data
    np.save("fused_eeg_data.npy", fused_data)
    logger.info("Fused EEG data saved.")

if __name__ == "__main__":
    main()