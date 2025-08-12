import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import os
import json
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'model': 'resnet50',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'input_size': 224,
    'output_size': 1000
}

class ModelType(Enum):
    RESNET50 = 'resnet50'
    VGG16 = 'vgg16'
    MOBILENET = 'mobilenet'

class Config:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return DEFAULT_CONFIG

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class VGG16(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MobileNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((self.config.config['input_size'], self.config.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        image = self.transform(image)
        return image

class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)

    def load_dataset(self, dataset_path: str) -> DataLoader:
        dataset = ImageDataset(dataset_path, self.data_processor)
        return DataLoader(dataset, batch_size=self.config.config['batch_size'], shuffle=True)

class ImageDataset(Dataset):
    def __init__(self, dataset_path: str, data_processor: DataProcessor):
        self.dataset_path = dataset_path
        self.data_processor = data_processor
        self.images = os.listdir(dataset_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = os.path.join(self.dataset_path, self.images[index])
        image = self.data_processor.process_image(image_path)
        label = int(self.images[index].split('.')[0])
        return image, label

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = self.load_model()

    def load_model(self) -> BaseModel:
        model_type = self.config.config['model']
        if model_type == ModelType.RESNET50.value:
            return ResNet50()
        elif model_type == ModelType.VGG16.value:
            return VGG16()
        elif model_type == ModelType.MOBILENET.value:
            return MobileNet()
        else:
            raise ValueError('Invalid model type')

    def train_model(self, dataset_loader: DatasetLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.config['learning_rate'], momentum=self.config.config['momentum'], weight_decay=self.config.config['weight_decay'])
        for epoch in range(self.config.config['epochs']):
            for batch in dataset_loader.load_dataset(self.config.config['dataset_path']):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                logger.info(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}')

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.pth')

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.model = self.load_model()

    def load_model(self) -> BaseModel:
        self.model.load_state_dict(torch.load('model.pth'))
        return self.model

    def evaluate_model(self, dataset_loader: DatasetLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataset_loader.load_dataset(self.config.config['dataset_path']):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        logger.info(f'Accuracy: {accuracy:.4f}')

def main():
    config = Config()
    dataset_loader = DatasetLoader(config)
    model_trainer = ModelTrainer(config)
    model_trainer.train_model(dataset_loader)
    model_trainer.save_model()
    model_evaluator = ModelEvaluator(config)
    model_evaluator.evaluate_model(dataset_loader)

if __name__ == '__main__':
    main()