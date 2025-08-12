import os
import sys
import logging
import setuptools
from setuptools import setup, find_packages
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define constants
PROJECT_NAME = "computer_vision"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Enhanced AI project based on cs.LG_2508.08216v1_Cross-Subject-and-Cross-Montage-EEG-Transfer-Learn with content analysis"

# Define dependencies
DEPENDENCIES = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
]

# Define setup function
def setup_package():
    try:
        # Create log file
        log_file = os.path.join(os.path.dirname(__file__), "setup.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set up package
        setup(
            name=PROJECT_NAME,
            version=PROJECT_VERSION,
            description=PROJECT_DESCRIPTION,
            long_description=open("README.md").read(),
            long_description_content_type="text/markdown",
            author="Your Name",
            author_email="your@email.com",
            url="https://github.com/your-username/computer_vision",
            packages=find_packages(),
            install_requires=DEPENDENCIES,
            include_package_data=True,
            zip_safe=False,
        )

        # Log success
        logging.info(f"Package {PROJECT_NAME} installed successfully.")

    except Exception as e:
        # Log error
        logging.error(f"Error installing package: {str(e)}")

        # Raise exception
        raise e

# Run setup function
if __name__ == "__main__":
    setup_package()