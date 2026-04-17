"""
Model Training Module
=====================
Trains and evaluates multiple ML models:
- Logistic Regression
- Random Forest
- XGBoost

Each model is cross-validated, evaluated, and saved to disk.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def setup_model_directory(base_path='models'):
    """
    Create and return model directory path.
    
    Args:
        base_path (str): Base directory for models
        
    Returns:
        str: Full path to models directory
    """
    os.makedirs(base_path, exist_ok=True)
    logger.info(f"Models directory ready: {base_path}")
    return os.path.abspath(base_path)


def save_model(model, model_name, model_dir='models'):
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model object
        model_name (str): Name for the model file
        model_dir (str): Directory to save model
        
    Returns:
        str: Path to saved model
    """
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_dir}/{model_name}_{timestamp}.pkl"
    
    joblib.dump(model, filename)
    logger.info(f"Model saved: {filename}")
    return filename


def setup_cross_validation(n_splits=5, random_state=42):
    """
    Setup stratified k-fold cross-validation.
    
    Args:
        n_splits (int): Number of folds
        random_state (int): Random seed
        
    Returns:
        StratifiedKFold: Cross-validator object
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info(f"Cross-validation setup: {n_splits}-fold")
    return cv


class LogisticRegressionModel:
    """Logistic Regression Model wrapper."""
    
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, class_weight='balanced', random_state=random_state)
        self.name = "Logistic Regression"
        self.scores = None
        
    def train_with_cv(self, pipeline, X_train, y_train, cv):
        """Train with cross-validation."""
        logger.info(f"Training {self.name} with cross-validation...")
        
        self.scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        logger.info(f"F1 scores: {self.scores}")
        logger.info(f"Mean F1: {self.scores.mean():.4f} (+/- {self.scores.std():.4f})")
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def evaluate(self, pipeline, X_test, y_test):
        """Evaluate on test set."""
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"{self.name} - Test Set Results")
        print(f"{'='*50}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"\n{cr}")
        
        return y_pred, cm, cr
    
    def save_model(self, pipeline, model_dir='models'):
        """Save trained pipeline to disk."""
        saved_path = save_model(pipeline, f"logistic_regression_pipeline", model_dir)
        logger.info(f"{self.name} saved to {saved_path}")
        return saved_path


class RandomForestModel:
    """Random Forest Model wrapper."""
    
    def __init__(self, n_estimators=200, max_depth=10, min_samples_split=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "Random Forest"
        self.scores = None
        
    def train_with_cv(self, pipeline, X_train, y_train, cv):
        """Train with cross-validation."""
        logger.info(f"Training {self.name} with cross-validation...")
        
        self.scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        logger.info(f"F1 scores: {self.scores}")
        logger.info(f"Mean F1: {self.scores.mean():.4f} (+/- {self.scores.std():.4f})")
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def evaluate(self, pipeline, X_test, y_test):
        """Evaluate on test set."""
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"{self.name} - Test Set Results")
        print(f"{'='*50}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"\n{cr}")
        
        return y_pred, cm, cr
    
    def save_model(self, pipeline, model_dir='models'):
        """Save trained pipeline to disk."""
        saved_path = save_model(pipeline, f"random_forest_pipeline", model_dir)
        logger.info(f"{self.name} saved to {saved_path}")
        return saved_path


class XGBoostModel:
    """XGBoost Model wrapper."""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric='logloss',
            verbosity=0,
            random_state=42
        )
        self.name = "XGBoost"
        self.scores = None
        
    def prepare_data(self, preprocessor, X_train, X_test, y_train, y_test):
        """Preprocess data for XGBoost training."""
        logger.info("Preprocessing data for XGBoost...")
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Additional validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_processed,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        
        logger.info(f"Training data shape: {X_tr.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test_processed.shape}")
        
        return X_tr, X_val, X_test_processed, y_tr, y_val
    
    def train_with_cv(self, X_train, y_train, cv):
        """Train with cross-validation."""
        logger.info(f"Training {self.name} with cross-validation...")
        
        self.scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
        logger.info(f"F1 scores: {self.scores}")
        logger.info(f"Mean F1: {self.scores.mean():.4f} (+/- {self.scores.std():.4f})")
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_with_validation(self, X_train, X_val, y_train, y_val):
        """Train with validation set for early stopping."""
        logger.info("Training with validation set...")
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"{self.name} - Test Set Results")
        print(f"{'='*50}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"\n{cr}")
        
        return y_pred, cm, cr
    
    def save_model(self, model_dir='models'):
        """Save trained model to disk."""
        saved_path = save_model(self.model, f"xgboost_model", model_dir)
        logger.info(f"{self.name} saved to {saved_path}")
        return saved_path


def train_all_models(prep_data, model_dir='models'):
    """
    Train all models and compare results.
    
    Args:
        prep_data (dict): Prepared data from data_preparation module
        model_dir (str): Directory to save trained models
        
    Returns:
        dict: Results from all models
    """
    X_train = prep_data['X_train']
    X_test = prep_data['X_test']
    y_train = prep_data['y_train']
    y_test = prep_data['y_test']
    preprocessor = prep_data['preprocessor']
    
    # Setup model directory
    setup_model_directory(model_dir)
    
    cv = setup_cross_validation()
    results = {}
    
    # ===== Logistic Regression =====
    lr_model = LogisticRegressionModel()
    from data_preparation import build_full_pipeline
    lr_pipeline = build_full_pipeline(lr_model.model, preprocessor)
    lr_pipeline = lr_model.train_with_cv(lr_pipeline, X_train, y_train, cv)
    lr_y_pred, lr_cm, lr_cr = lr_model.evaluate(lr_pipeline, X_test, y_test)
    lr_model_path = lr_model.save_model(lr_pipeline, model_dir)
    results['logistic_regression'] = {
        'model': lr_model,
        'pipeline': lr_pipeline,
        'predictions': lr_y_pred,
        'confusion_matrix': lr_cm,
        'model_path': lr_model_path,
        'f1_score': lr_model.scores.mean()
    }
    
    # ===== Random Forest =====
    rf_model = RandomForestModel()
    rf_pipeline = build_full_pipeline(rf_model.model, preprocessor)
    rf_pipeline = rf_model.train_with_cv(rf_pipeline, X_train, y_train, cv)
    rf_y_pred, rf_cm, rf_cr = rf_model.evaluate(rf_pipeline, X_test, y_test)
    rf_model_path = rf_model.save_model(rf_pipeline, model_dir)
    results['random_forest'] = {
        'model': rf_model,
        'pipeline': rf_pipeline,
        'predictions': rf_y_pred,
        'confusion_matrix': rf_cm,
        'model_path': rf_model_path,
        'f1_score': rf_model.scores.mean()
    }
    
    # ===== XGBoost =====
    xgb_model = XGBoostModel()
    X_tr, X_val, X_test_proc, y_tr, y_val = xgb_model.prepare_data(
        preprocessor, X_train, X_test, y_train, y_test
    )
    xgb_model.train_with_cv(X_tr, y_tr, cv)
    xgb_model.train_with_validation(X_tr, X_val, y_tr, y_val)
    xgb_y_pred, xgb_cm, xgb_cr = xgb_model.evaluate(X_test_proc, y_test)
    xgb_model_path = xgb_model.save_model(model_dir)
    results['xgboost'] = {
        'model': xgb_model,
        'predictions': xgb_y_pred,
        'confusion_matrix': xgb_cm,
        'model_path': xgb_model_path,
        'f1_score': xgb_model.scores.mean()
    }
    
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"Logistic Regression Mean F1: {lr_model.scores.mean():.4f}")
    print(f"Random Forest Mean F1: {rf_model.scores.mean():.4f}")
    print(f"XGBoost Mean F1: {xgb_model.scores.mean():.4f}")
    print(f"\nAll models saved to: {os.path.abspath(model_dir)}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Model training module loaded successfully.")
