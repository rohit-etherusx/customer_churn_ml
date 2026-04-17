# Customer Churn Prediction - Project Guide

## 📋 Quick Start

### Installation & Setup
```bash
# Navigate to project directory
cd /home/rohit/projects/customer_churn

# Activate virtual environment
source ml-env/bin/activate

# Install Streamlit (if not already installed)
pip install streamlit

# Run the dashboard
streamlit run main.py
```

### Access the Dashboard
- Open browser and go to: `http://localhost:8501`
- The dashboard will load with two phases

---

## 🏗️ Project Structure Overview

```
customer_churn/
│
├── 📄 main.py                          # Main Streamlit dashboard (START HERE)
│   ├─ Phase 1: Business Prediction     # Real-time churn predictions
│   └─ Phase 2: Model Metrics          # Performance analysis
│
├── 🗂️ src/                            # Core modules
│   ├─ data_download.py                 # Download dataset from Kaggle
│   ├─ eda.py                          # Exploratory Data Analysis
│   ├─ data_preparation.py             # Feature preprocessing
│   ├─ model_training.py               # ML model training
│   ├─ pipeline.py                     # Complete pipeline orchestration
│   └─ __init__.py                     # Package initialization
│
├── 📊 models/                         # Saved trained models
│   ├─ logistic_regression_pipeline_*.pkl
│   ├─ random_forest_pipeline_*.pkl
│   └─ xgboost_model_*.pkl             # Used by Phase 1
│
├── 📈 notebooks/
│   ├─ analysis.ipynb                  # Original analysis notebook
│   └─ modeling.ipynb                  # (Empty - use src/ instead)
│
├── 📁 data/
│   └─ synthetic_customer_churn_100k.csv  # 100k customer records
│
├── 🔄 scripts/
│   └─ (Legacy scripts - use src/ instead)
│
├── 📄 README.md                        # Project documentation
├── 📄 INSTRUCTIONS.md                  # This file
├── 📄 .gitignore                       # Git ignore configuration
└── 📄 CODEBASE_SANITY_CHECK.log       # Change tracking log

```

---

## 🚀 How to Use the Project

### **Option 1: Run Dashboard (Recommended)**
```bash
streamlit run main.py
```
**Use this to:**
- Make real-time predictions on customer data
- View model performance metrics
- Get business recommendations

---

### **Option 2: Run Complete Pipeline**
```bash
cd src/
python pipeline.py
```
**This will:**
1. Load and explore data (EDA)
2. Prepare features and split data
3. Train all 3 models
4. Save trained models to `/models/`

---

### **Option 3: Run Individual Modules**

#### **1. Exploratory Data Analysis (EDA)**
```python
from src.eda import run_eda
df = run_eda('data/synthetic_customer_churn_100k.csv')
```
**Output:** Dataset overview, churn distribution, feature analysis

#### **2. Data Preparation**
```python
from src.data_preparation import prepare_data_pipeline
prep_data = prepare_data_pipeline(df)
```
**Output:** Train/test split, preprocessor, feature types

#### **3. Model Training**
```python
from src.model_training import train_all_models
results = train_all_models(prep_data, model_dir='models')
```
**Output:** Trained models saved to `/models/` with metrics

---

## 📊 Dashboard Guide

### **Phase 1: Business Analysis & Prediction**

**Purpose:** Real-time churn prediction for business decision-making

**Features:**
- 📥 **Input Method Selection**
  - Manual Entry: Fill customer details manually
  - Upload CSV: Batch process customer records
  
- 🔮 **Prediction Metrics**
  - Churn Status (Will Stay / Will Churn)
  - Risk Level (Low/Medium/High)
  - Confidence Score
  
- 💼 **Business Recommendations**
  - Retention actions for high-risk customers
  - Maintenance actions for satisfied customers
  - Specific next steps and timelines

**Model Used:** XGBoost (best performing model)

**Customer Input Fields:**
- Age, Gender, Tenure (months)
- Monthly Charges, Total Charges
- Contract Type
- Internet Service Type
- Online Security Status
- Tech Support Status

**Output:**
```
Prediction Result:
├─ Churn Status: Will Stay / Will Churn
├─ Risk Level: 🟢 Low / 🟡 Medium / 🔴 High
├─ Churn Probability: X.XX%
├─ Confidence: X.XX%
└─ Recommendations: [Specific actions]
```

---

### **Phase 2: Model Statistics & Metrics**

**Purpose:** Evaluate model performance and understand predictions

**Tab 1: Overview 📊**
- Model information (type, creation date, samples)
- Dataset statistics (total records, features, churn rate)
- Churn distribution chart

**Tab 2: Performance 🎯**
- **Key Metrics:**
  - Accuracy: Correct predictions percentage
  - Precision: Correct churn predictions / all churn predictions
  - Recall: Caught churners / all actual churners
  - F1 Score: Balance between Precision and Recall
  - ROC AUC: Model's discrimination ability
  
- **Confusion Matrix:**
  - True Positives (TP): Correctly predicted churners
  - True Negatives (TN): Correctly predicted stayers
  - False Positives (FP): Wrongly flagged for churn
  - False Negatives (FN): Missed churners
  
- **Classification Report:**
  - Per-class metrics (Precision, Recall, F1)
  - Weighted averages

**Tab 3: Features 🔍**
- Top 20 most important features
- Feature importance scores
- Bar chart visualization
- Top feature highlighted

**Tab 4: Analysis 📉**
- Model strengths and insights
- Precision vs Recall trade-offs
- Business impact analysis
- Recommendations for improvement
- Model maintenance guidelines

---

## 📂 Core Module Reference

### **src/data_download.py**
```python
from src.data_download import download_dataset
download_dataset(destination='data')
```
- Downloads customer churn data from Kaggle
- Extracts to specified directory
- Logs progress

---

### **src/eda.py**
```python
from src.eda import run_eda
df = run_eda('data/synthetic_customer_churn_100k.csv')
```
**Functions:**
- `load_data()` - Load CSV file
- `basic_overview()` - Dataset shape, dtypes, missing values
- `churn_analysis()` - Churn distribution and relationships
- `feature_analysis()` - Identify key features
- `data_quality_check()` - Data integrity validation
- `run_eda()` - Complete EDA pipeline

**Output:** Printed analysis and DataFrame

---

### **src/data_preparation.py**
```python
from src.data_preparation import prepare_data_pipeline
prep_data = prepare_data_pipeline(df)
```
**Functions:**
- `prepare_target_features()` - Separate X, y
- `identify_feature_types()` - Categorical vs Numerical
- `build_preprocessor()` - Scaling and encoding pipeline
- `train_test_split_data()` - 80/20 stratified split
- `prepare_data_pipeline()` - Complete preparation

**Output Dictionary:**
```python
{
    'X': features,
    'y': target,
    'X_train': training features,
    'X_test': test features,
    'y_train': training target,
    'y_test': test target,
    'preprocessor': fitted preprocessor,
    'categorical_features': list,
    'numeric_features': list
}
```

---

### **src/model_training.py**
```python
from src.model_training import train_all_models
results = train_all_models(prep_data, model_dir='models')
```
**Classes:**
- `LogisticRegressionModel` - Logistic Regression wrapper
- `RandomForestModel` - Random Forest wrapper
- `XGBoostModel` - XGBoost wrapper

**Functions:**
- `setup_cross_validation()` - 5-fold stratified CV
- `save_model()` - Save with timestamp versioning
- `setup_model_directory()` - Create models folder
- `train_all_models()` - Train all 3 models and compare

**Output Dictionary:**
```python
{
    'logistic_regression': {
        'model': model_obj,
        'pipeline': fitted_pipeline,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'model_path': '/models/...',
        'f1_score': 0.xxx
    },
    'random_forest': {...},
    'xgboost': {...}
}
```

---

### **src/pipeline.py**
```python
python src/pipeline.py
```
**Main Orchestration:**
- Loads data
- Runs EDA
- Prepares features
- Trains models
- Saves results

**Execution Flow:**
```
1. Load dataset
   ↓
2. Exploratory Data Analysis
   ↓
3. Feature Preparation
   ↓
4. Train Models (LR, RF, XGB)
   ↓
5. Save Models to /models/
```

---

## 📊 Data Files

### **synthetic_customer_churn_100k.csv**
- 100,000 customer records
- Columns: Customer ID, Age, Gender, Tenure, Charges, Services, Churn
- Churn Rate: ~26% (represents typical telecom data)
- Used by all modules

### **Model Files (in /models/)**
- Timestamped filenames for versioning
- Format: `{model_name}_{YYYYMMDD_HHMMSS}.pkl`
- Loaded automatically by dashboard

---

## 🔄 Workflow Examples

### **Example 1: Make a Prediction**
```bash
streamlit run main.py
# Go to Phase 1 → Manual Entry
# Fill in customer details
# Click "Predict Churn Risk"
```

### **Example 2: Check Model Performance**
```bash
streamlit run main.py
# Go to Phase 2 → Performance Tab
# View accuracy, precision, recall, F1 score
```

### **Example 3: Identify Important Features**
```bash
streamlit run main.py
# Go to Phase 2 → Features Tab
# View top 20 most important features
```

### **Example 4: Retrain Models**
```bash
source ml-env/bin/activate
python src/pipeline.py
# Models will be retrained and saved to /models/
```

---

## 🛠️ Configuration

### **Model Parameters**
Edit `src/model_training.py` to adjust:
- XGBoost: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
- Random Forest: `n_estimators=200`, `max_depth=10`
- Logistic Regression: `max_iter=1000`

### **Data Parameters**
Edit `src/data_preparation.py`:
- Test split: `test_size=0.2` (80/20 split)
- Random seed: `random_state=42` (reproducibility)
- Cross-validation folds: `n_splits=5`

---

## 📈 Performance Metrics Explained

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Cost of false alarms |
| **Recall** | TP / (TP + FN) | Cost of missed churners |
| **F1 Score** | 2 * (Precision * Recall) / (P + R) | Balanced metric |
| **ROC AUC** | Area under ROC curve | Discrimination ability |
| **Specificity** | TN / (TN + FP) | True negative rate |

---

## 🐛 Troubleshooting

### **Issue: "No XGBoost model found"**
```bash
# Solution: Run the pipeline first
python src/pipeline.py
```

### **Issue: "ImportError: No module named 'src'"**
```bash
# Solution: Run from project root
cd /home/rohit/projects/customer_churn
streamlit run main.py
```

### **Issue: "Column not found in preprocessor"**
```bash
# Solution: Ensure input data has same columns as training data
# Check column names in data_preparation.py
```

### **Issue: "Model accuracy is low"**
```bash
# Solutions:
# 1. Check data quality in Phase 2 → Overview
# 2. Retrain with fresh data: python src/pipeline.py
# 3. Adjust hyperparameters in model_training.py
```

---

## 📚 Additional Resources

### **Files to Review:**
- `README.md` - Project overview
- `CODEBASE_SANITY_CHECK.log` - Change history
- `.gitignore` - Git configuration

### **Key Directories:**
- `src/` - All processing modules
- `models/` - Saved trained models
- `data/` - Input datasets
- `notebooks/` - Legacy analysis

---

## 🎯 Next Steps

1. **First Time?** Run the pipeline:
   ```bash
   python src/pipeline.py
   ```

2. **Make Predictions?** Start the dashboard:
   ```bash
   streamlit run main.py
   ```

3. **Analyze Models?** Check Phase 2 metrics

4. **Improve Models?** Adjust parameters and retrain

5. **Deploy?** Use Phase 1 dashboard in production

---

## 👥 Support & Questions

For issues or improvements:
1. Check `CODEBASE_SANITY_CHECK.log` for recent changes
2. Review module docstrings in `src/`
3. Consult this guide's troubleshooting section

---

**Last Updated:** 2026-04-17
**Version:** 1.0.0
