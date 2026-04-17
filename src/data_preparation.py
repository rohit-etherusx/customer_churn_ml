"""
Data Preparation Module
=======================
Handles data preprocessing, feature engineering, and pipeline setup.
Includes data splitting, scaling, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def prepare_target_features(df):
    """
    Separate features and target variable.
    Drops CustomerID (just an ID, not a predictor) and selects only useful features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    logger.info("Preparing features and target...")
    
    # Drop CustomerID and Churn
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def identify_feature_types(X):
    """
    Identify categorical and numerical features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        
    Returns:
        tuple: (categorical_features, numerical_features)
    """
    logger.info("Identifying feature types...")
    
    categorical_feat = X.select_dtypes(include=['object']).columns
    numeric_feat = X.select_dtypes(include=['float', 'int64']).columns
    
    logger.info(f"Categorical features ({len(categorical_feat)}): {list(categorical_feat)}")
    logger.info(f"Numerical features ({len(numeric_feat)}): {list(numeric_feat)}")
    
    return categorical_feat, numeric_feat


def build_preprocessor(categorical_features, numeric_features):
    """
    Build preprocessing pipeline with scaling and encoding.
    
    Args:
        categorical_features: List of categorical column names
        numeric_features: List of numerical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    logger.info("Building preprocessing pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    return preprocessor


def build_full_pipeline(model, preprocessor):
    """
    Build complete pipeline with preprocessor and model.
    
    Args:
        model: Scikit-learn estimator
        preprocessor: ColumnTransformer
        
    Returns:
        Pipeline: Complete pipeline
    """
    logger.info("Building full pipeline...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Handle class imbalance
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Training target distribution:\n{y_train.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def prepare_data_pipeline(df):
    """
    Execute complete data preparation pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing all prepared components
    """
    # Prepare features and target
    X, y = prepare_target_features(df)
    
    # Identify feature types
    categorical_feat, numeric_feat = identify_feature_types(X)
    
    # Build preprocessor
    preprocessor = build_preprocessor(categorical_feat, numeric_feat)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'categorical_features': categorical_feat,
        'numeric_features': numeric_feat
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, '/home/rohit/projects/customer_churn')
    
    from src.eda import load_data
    df = load_data('../data/synthetic_customer_churn_100k.csv')
    prep_data = prepare_data_pipeline(df)
    print("\nData preparation complete!")
