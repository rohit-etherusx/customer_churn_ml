"""
Customer Churn Prediction - Main Pipeline
==========================================
Orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute complete ML pipeline."""
    
    try:
        logger.info("="*60)
        logger.info("CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load and explore data
        logger.info("\n[STEP 1] Exploratory Data Analysis")
        from src.eda import run_eda
        df = run_eda('data/synthetic_customer_churn_100k.csv')
        
        # Step 2: Prepare data
        logger.info("\n[STEP 2] Data Preparation")
        from src.data_preparation import prepare_data_pipeline
        prep_data = prepare_data_pipeline(df)
        
        # Step 3: Train models
        logger.info("\n[STEP 3] Model Training & Evaluation")
        from src.model_training import train_all_models
        results = train_all_models(prep_data)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
