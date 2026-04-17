"""
Exploratory Data Analysis Module
=================================
Performs comprehensive exploratory data analysis on customer churn dataset.
Includes data overview, distribution analysis, and feature relationships.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Load dataset from CSV file.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data shape: {df.shape}")
    return df


def basic_overview(df):
    """
    Print basic dataset information.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "="*50)
    print("BASIC DATASET OVERVIEW")
    print("="*50)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 10 rows:\n{df.head(10)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isna().sum()}")


def churn_analysis(df):
    """
    Analyze churn distribution and relationships.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "="*50)
    print("CHURN ANALYSIS")
    print("="*50)
    
    # Churn distribution
    print(f"\nChurn Value Counts:\n{df['Churn'].value_counts(normalize=False)}")
    
    # Contract vs Churn
    print(f"\nContract vs Churn (normalized by Contract):\n{pd.crosstab(df['Contract'], df['Churn'], normalize='index')}")
    
    # MonthlyCharges vs Churn
    print(f"\nMonthly Charges Statistics by Churn:\n{df.groupby('Churn')['MonthlyCharges'].describe()}")
    
    # Tenure vs Churn
    print(f"\nAverage Tenure by Churn Status:\n{df.groupby('Churn')['Tenure'].mean()}")


def feature_analysis(df):
    """
    Identify and display key features for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "="*50)
    print("KEY FEATURES FOR MODELING")
    print("="*50)
    
    key_features = [
        'Age',
        'Gender', 
        'Tenure',
        'MonthlyCharges',
        'Contract',
        'TotalCharges',
        'Churn'
    ]
    
    print("\nKey Features Identified:")
    for i, feature in enumerate(key_features, 1):
        print(f"  {i}. {feature}")
    
    return key_features


def data_quality_check(df):
    """
    Perform data quality checks.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Quality check results
    """
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    checks = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numerical_cols': len(df.select_dtypes(include=['float', 'int64']).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns)
    }
    
    for check, value in checks.items():
        print(f"{check}: {value}")
    
    return checks


def run_eda(filepath):
    """
    Execute complete EDA pipeline.
    
    Args:
        filepath (str): Path to CSV file
    """
    df = load_data(filepath)
    basic_overview(df)
    churn_analysis(df)
    feature_analysis(df)
    data_quality_check(df)
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run_eda('../data/synthetic_customer_churn_100k.csv')
