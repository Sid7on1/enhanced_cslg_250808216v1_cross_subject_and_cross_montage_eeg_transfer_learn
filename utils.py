import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config import PROJECT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define global constants
TEMP_DIR = tempfile.mkdtemp()

# Define exception classes
class InvalidInputError(Exception):
    """Exception raised for errors in input validation."""
    pass

class AlgorithmError(Exception):
    """Exception raised for errors during algorithm execution."""
    pass

# Helper functions
def validate_input(input_data: Any) -> None:
    """
    Validate the input data.

    Args:
        input_data (Any): Data to be validated.

    Raises:
        InvalidInputError: If the input data is invalid.
    """
    if not isinstance(input_data, (list, np.ndarray, Tensor, pd.DataFrame, dict)):
        raise InvalidInputError("Invalid input type. Expected list, numpy array, torch Tensor, pandas DataFrame, or dict.")

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    config_path = os.path.join(PROJECT_DIR, config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        config = pd.read_csv(config_path)
        return config.set_index('key')['value'].to_dict()
    except pd.errors.EmptyDataError:
        raise ValueError("Configuration file is empty.")

# Main utility class
class Utils:
    """
    Utility functions for the project.

    Attributes:
        config (Dict[str, Any]): Project configuration.
    """
    config: Dict[str, Any]

    def __init__(self, config_file: str = "config.csv"):
        """
        Initialize the utility class with project configuration.

        Args:
            config_file (str, optional): Path to the configuration file. Defaults to "config.csv".
        """
        self.config = load_config(config_file)

    @staticmethod
    def preprocess_data(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess the input data.

        Args:
            data (Union[List[Dict[str, Any]], pd.DataFrame]): Input data as a list of dictionaries or a pandas DataFrame.

        Returns:
            pd.DataFrame: Preprocessed data as a pandas DataFrame.

        Raises:
            InvalidInputError: If the input data is not valid.
        """
        validate_input(data)
        if isinstance(data, list):
            data = pd.DataFrame(data)

        # Perform data preprocessing steps as per project requirements
        # Example: Handling missing values, feature engineering, etc.
        data['feature_1'] = data['feature_1'].fillna(data['feature_1'].mean())
        data['feature_2'] = data['feature_2'].apply(lambda x: x * 2)

        return data

    @staticmethod
    def save_model(model: torch.nn.Module, model_path: str) -> None:
        """
        Save a PyTorch model to a file.

        Args:
            model (torch.nn.Module): Model to be saved.
            model_path (str): Path to save the model.

        Raises:
            AlgorithmError: If an error occurs during model saving.
        """
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise AlgorithmError("Failed to save model.")

    @staticmethod
    def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from a file.

        Args:
            model (torch.nn.Module): Model to load the weights into.
            model_path (str): Path to the saved model weights.

        Returns:
            torch.nn.Module: Loaded model.

        Raises:
            FileNotFoundError: If the model file is not found.
            AlgorithmError: If an error occurs during model loading.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        try:
            model.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise AlgorithmError("Failed to load model.")

    def algorithm_velocity_threshold(self, data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Implement the velocity-threshold algorithm as described in the research paper.

        Args:
            data (pd.DataFrame): Input data containing velocity information.
            threshold (float, optional): Velocity threshold value. Defaults to 0.5.

        Returns:
            pd.DataFrame: Processed data with velocity threshold applied.

        Raises:
            InvalidInputError: If the input data is not valid.
        """
        validate_input(data)
        if 'velocity' not in data.columns:
            raise InvalidInputError("Input data must contain a 'velocity' column.")

        # Apply velocity threshold as per the paper's methodology
        data['velocity_threshold'] = data['velocity'].apply(lambda x: min(x, threshold))
        data['output'] = data['velocity_threshold'] * self.config['velocity_factor']

        return data

    # Additional utility functions and methods can be added here

    # Example: Implementing the Flow Theory algorithm
    # def algorithm_flow_theory(data: pd.DataFrame) -> pd.DataFrame:
    #     ...

# Example usage
if __name__ == "__main__":
    # Create utility instance
    utils = Utils()

    # Preprocess data
    input_data = [
        {"feature_1": 1, "feature_2": 2},
        {"feature_1": 3, "feature_2": 4},
        {"feature_1": 5, "feature_2": 6}
    ]
    preprocessed_data = utils.preprocess_data(input_data)
    print(preprocessed_data)

    # Apply velocity-threshold algorithm
    velocity_data = pd.DataFrame({
        'velocity': [0.2, 0.6, 0.8, 0.4, 0.3]
    })
    processed_velocity_data = utils.algorithm_velocity_threshold(velocity_data)
    print(processed_velocity_data)