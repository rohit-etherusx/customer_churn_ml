"""
Data Download Module
=====================
Downloads customer churn dataset from Kaggle using kagglehub.
This script handles dataset acquisition and organization.
"""

import kagglehub
import shutil
import os
import logging

logger = logging.getLogger(__name__)


def download_dataset(destination="../data"):
    """
    Download the telco customer churn dataset from Kaggle.
    
    Args:
        destination (str): Path where dataset will be saved
        
    Returns:
        str: Path to the dataset destination
    """
    try:
        logger.info("Starting dataset download from Kaggle...")
        
        # Download dataset
        path = kagglehub.dataset_download("dhrubangtalukdar/telco-customer-churn-data")
        
        # Create destination directory
        os.makedirs(destination, exist_ok=True)
        
        # Copy files to destination
        for file in os.listdir(path):
            src_file = os.path.join(path, file)
            dst_file = os.path.join(destination, file)
            shutil.copy(src_file, dst_file)
            logger.info(f"Copied: {file}")
        
        abs_path = os.path.abspath(destination)
        logger.info(f"Dataset successfully downloaded to: {abs_path}")
        return abs_path
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_dataset()
