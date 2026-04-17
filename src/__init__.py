"""
Customer Churn Prediction Package
==================================
Package for end-to-end customer churn prediction modeling.

Modules:
- data_download: Dataset acquisition from Kaggle
- eda: Exploratory data analysis
- data_preparation: Feature engineering and data preprocessing
- model_training: Model training and evaluation
- main: Pipeline orchestration
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from src.eda import run_eda, load_data
from src.data_preparation import prepare_data_pipeline, build_preprocessor
from src.model_training import train_all_models

__all__ = [
    'run_eda',
    'load_data',
    'prepare_data_pipeline',
    'build_preprocessor',
    'train_all_models'
]
