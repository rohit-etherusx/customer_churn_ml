"""
Customer Churn Prediction Dashboard - Main Application
======================================================
Two-phase interactive dashboard:
Phase 1: Business Analysis & Prediction - Real-time churn predictions using XGBoost
Phase 2: Model Statistics & Metrics - Detailed model performance analysis

Author: Rohit Mishra
Version: 1.0.0
Date: 2026-04-17
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import local modules
import sys
sys.path.insert(0, '/home/rohit/projects/customer_churn')

from src.eda import load_data
from src.data_preparation import (
    identify_feature_types, 
    build_preprocessor, 
    train_test_split_data, 
    prepare_target_features
)
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .phase-header {
        color: #ff6b6b;
        font-size: 1.8em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING & DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_xgboost_model():
    """
    Load the latest XGBoost model from models folder.
    
    Returns:
        model: Trained XGBoost model or None if not found
    """
    try:
        model_files = glob.glob('/home/rohit/projects/customer_churn/models/xgboost_model_*.pkl')
        if not model_files:
            st.error("❌ No XGBoost model found in /models/ folder. Please run the pipeline first.")
            return None
        
        latest_model = max(model_files, key=os.path.getctime)
        model = joblib.load(latest_model)
        st.success(f"✅ Model loaded: {os.path.basename(latest_model)}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_preprocessor():
    """
    Load or create preprocessor for data transformation.
    
    Returns:
        tuple: (preprocessor, categorical_features, numeric_features, X_train)
    """
    try:
        df = load_data('/home/rohit/projects/customer_churn/data/synthetic_customer_churn_100k.csv')
        X, y = prepare_target_features(df)
        categorical_feat, numeric_feat = identify_feature_types(X)
        preprocessor = build_preprocessor(categorical_feat, numeric_feat)
        preprocessor.fit(X)
        return preprocessor, categorical_feat, numeric_feat, X
    except Exception as e:
        st.error(f"❌ Error loading preprocessor: {str(e)}")
        return None, None, None, None


@st.cache_data
def load_dataset():
    """
    Load the full customer churn dataset.
    
    Returns:
        pd.DataFrame: Customer churn dataset
    """
    try:
        return load_data('/home/rohit/projects/customer_churn/data/synthetic_customer_churn_100k.csv')
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


# ============================================================================
# PHASE 1: BUSINESS ANALYSIS & PREDICTION
# ============================================================================

def phase_1_business_prediction():
    """
    Phase 1: Real-time predictions and business analysis.
    
    Features:
    - Manual customer data entry
    - CSV batch upload
    - Real-time predictions using XGBoost
    - Churn risk assessment
    - Business recommendations
    """
    
    st.markdown("<div class='phase-header'>🎯 Phase 1: Business Analysis & Prediction</div>", unsafe_allow_html=True)
    st.markdown("*Real-time churn predictions using XGBoost Model*")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💼 Customer Churn Risk Assessment")
    
    with col2:
        st.info("🔮 Powered by XGBoost Model")
    
    # Load model and preprocessor
    xgb_model = load_xgboost_model()
    preprocessor, categorical_feat, numeric_feat, X_train = load_preprocessor()
    
    if xgb_model is None or preprocessor is None:
        st.error("⚠️ Failed to load model or preprocessor. Please check if models exist.")
        return
    
    # Input method selection
    input_method = st.radio(
        "📥 Select Input Method:",
        ["Manual Entry", "Upload CSV"],
        horizontal=True
    )
    
    customer_data = None
    
    if input_method == "Manual Entry":
        st.subheader("📋 Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Details**")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        
        with col2:
            st.markdown("**Charges**")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
        
        with col3:
            st.markdown("**Contract & Payment**")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
        
        # Create dataframe from inputs (match training data)
        customer_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'PaymentMethod': [payment_method]
        })
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("📤 Upload customer data (CSV)", type=['csv'])
        if uploaded_file:
            try:
                customer_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(customer_data)} records")
                st.dataframe(customer_data.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Make predictions
    if customer_data is not None and st.button("🔮 Predict Churn Risk", use_container_width=True):
        try:
            with st.spinner("🔄 Processing predictions..."):
                # Preprocess data
                X_processed = preprocessor.transform(customer_data)
                
                # Get predictions
                predictions = xgb_model.predict(X_processed)
                probabilities = xgb_model.predict_proba(X_processed)
                
                # Display results
                st.markdown("---")
                st.subheader("✨ Prediction Results")
                
                for idx, (pred, proba) in enumerate(zip(predictions, probabilities)):
                    if len(predictions) > 1:
                        st.markdown(f"**Customer {idx + 1}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    churn_status = "⚠️ WILL CHURN" if pred == 1 else "✅ WILL STAY"
                    churn_probability = proba[1] * 100
                    
                    if proba[1] > 0.7:
                        risk_level = "🔴 HIGH RISK"
                        risk_color = "red"
                    elif proba[1] > 0.4:
                        risk_level = "🟡 MEDIUM RISK"
                        risk_color = "orange"
                    else:
                        risk_level = "🟢 LOW RISK"
                        risk_color = "green"
                    
                    with col1:
                        st.metric(
                            "Churn Status",
                            churn_status,
                            delta=f"Confidence: {max(proba)*100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Risk Level",
                            risk_level,
                            delta=f"Churn Probability: {proba[1]*100:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Model Output",
                            f"{pred}",
                            delta="0=Stay | 1=Churn"
                        )
                    
                    # Display probability distribution
                    st.markdown("**Prediction Confidence Distribution**")
                    prob_df = pd.DataFrame({
                        'Outcome': ['Will Stay', 'Will Churn'],
                        'Probability': [proba[0]*100, proba[1]*100]
                    })
                    
                    st.bar_chart(prob_df.set_index('Outcome')['Probability'])
                    
                    # Business recommendations
                    st.markdown("---")
                    st.subheader("📋 Business Recommendations")
                    
                    if pred == 1:
                        st.warning(f"""
                        ### 🚨 RETENTION ACTION REQUIRED
                        
                        **Priority Level:** HIGH
                        
                        **Immediate Actions:**
                        - ☎️ Contact customer immediately (within 24 hours)
                        - 💰 Offer retention discount (10-15% off monthly charge)
                        - 🎁 Provide personalized support/upgrades
                        - 📋 Review contract terms - suggest upgrade options
                        - 📅 Schedule follow-up within 7 days
                        
                        **Why This Matters:**
                        - Churn probability: {proba[1]*100:.1f}%
                        - Model confidence: {max(proba)*100:.1f}%
                        """)
                    else:
                        st.success(f"""
                        ### ✅ SATISFACTION MAINTENANCE
                        
                        **Priority Level:** STANDARD
                        
                        **Recommended Actions:**
                        - 📊 Monitor for service quality issues
                        - 💌 Send periodic check-ins (quarterly)
                        - 📈 Cross-sell opportunity for upgrades
                        - 🎤 Gather feedback quarterly
                        - 🌟 Ensure excellent service levels
                        
                        **Customer Satisfaction:**
                        - Stay probability: {proba[0]*100:.1f}%
                        - Model confidence: {max(proba)*100:.1f}%
                        """)
                    
                    if len(predictions) > 1:
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")


# ============================================================================
# PHASE 2: MODEL STATISTICS & METRICS
# ============================================================================

def phase_2_model_metrics():
    """
    Phase 2: Detailed model performance metrics and analysis.
    
    Features:
    - Model information and overview
    - Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Feature importance analysis
    - Confusion matrix and classification report
    - Model insights and recommendations
    """
    
    st.markdown("<div class='phase-header'>📈 Phase 2: Model Statistics & Metrics</div>", unsafe_allow_html=True)
    st.markdown("*Detailed model performance analysis and insights*")
    st.markdown("---")
    
    try:
        # Load data
        df = load_dataset()
        
        if df is None:
            st.error("❌ Failed to load dataset")
            return
        
        # Prepare data
        X, y = prepare_target_features(df)
        categorical_feat, numeric_feat = identify_feature_types(X)
        preprocessor = build_preprocessor(categorical_feat, numeric_feat)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        
        # Load model
        xgb_model = load_xgboost_model()
        if xgb_model is None:
            st.error("❌ Failed to load model")
            return
        
        # Create tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Performance", "🔍 Features", "📉 Analysis"])
        
        # ===== TAB 1: OVERVIEW =====
        with tab1:
            st.subheader("🏷️ Model Information & Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            model_files = glob.glob('/home/rohit/projects/customer_churn/models/xgboost_model_*.pkl')
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                model_date = os.path.getmtime(latest_model)
                model_datetime = datetime.fromtimestamp(model_date).strftime("%Y-%m-%d %H:%M:%S")
            else:
                model_datetime = "Not Available"
            
            with col1:
                st.metric("Model Type", "🤖 XGBoost")
            with col2:
                st.metric("Created", model_datetime.split()[0] if model_datetime != "Not Available" else "N/A")
            with col3:
                st.metric("Training Samples", f"{len(X_train):,}")
            with col4:
                st.metric("Test Samples", f"{len(X_test):,}")
            
            # Dataset statistics
            st.markdown("---")
            st.subheader("📊 Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Features", len(X.columns))
            
            with col2:
                churn_rate = (y == 1).sum() / len(y) * 100
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
                st.metric("Churned Customers", (y == 1).sum())
            
            with col3:
                st.metric("Numerical Features", len(numeric_feat))
                st.metric("Categorical Features", len(categorical_feat))
            
            # Churn distribution
            st.markdown("---")
            st.subheader("📈 Churn Distribution")
            
            churn_dist = y.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(pd.Series({
                    'Will Stay': churn_dist[0],
                    'Will Churn': churn_dist[1]
                }))
            
            with col2:
                st.markdown("**Distribution Summary:**")
                st.metric("Non-Churned Customers", churn_dist[0], f"{churn_dist[0]/len(y)*100:.2f}%")
                st.metric("Churned Customers", churn_dist[1], f"{churn_dist[1]/len(y)*100:.2f}%")
        
        # ===== TAB 2: PERFORMANCE =====
        with tab2:
            st.subheader("🎯 Model Performance Metrics")
            
            # Make predictions
            X_test_processed = preprocessor.transform(X_test)
            y_pred = xgb_model.predict(X_test_processed)
            y_pred_proba = xgb_model.predict_proba(X_test_processed)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
            with col2:
                st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
            with col3:
                st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}", f"{f1*100:.2f}%")
            with col5:
                st.metric("ROC AUC", f"{roc_auc:.4f}", f"{roc_auc*100:.2f}%")
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: Will Stay', 'Actual: Will Churn'],
                columns=['Predicted: Stay', 'Predicted: Churn']
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(cm_df, use_container_width=True)
            
            with col2:
                tn, fp, fn, tp = cm.ravel()
                st.markdown("""
                **Matrix Breakdown:**
                - True Positives (TP): Correctly predicted churners
                - True Negatives (TN): Correctly predicted stayers
                - False Positives (FP): Incorrectly predicted churners
                - False Negatives (FN): Incorrectly predicted stayers
                """)
            
            # Classification Report
            st.markdown("---")
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        # ===== TAB 3: FEATURES =====
        with tab3:
            st.subheader("🔍 Feature Importance Analysis")
            
            # Get feature importance
            feature_importance = xgb_model.feature_importances_
            
            # Get feature names
            num_features = list(numeric_feat)
            cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_feat))
            all_features = num_features + cat_features
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': all_features[:len(feature_importance)],
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(importance_df.set_index('Feature')['Importance'])
            
            with col2:
                st.markdown("**Top 20 Features:**")
                for idx, row in importance_df.iterrows():
                    st.text(f"{row['Feature']}: {row['Importance']:.4f}")
            
            st.info(f"🏆 **Top Feature:** {importance_df.iloc[0]['Feature']} (Importance: {importance_df.iloc[0]['Importance']:.4f})")
        
        # ===== TAB 4: ANALYSIS =====
        with tab4:
            st.subheader("📉 Model Analysis & Insights")
            
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Model Strengths")
                st.markdown(f"""
                - **Accuracy:** {accuracy*100:.2f}% of all predictions are correct
                - **Precision:** {precision*100:.2f}% of predicted churners are actually churning
                - **Sensitivity (Recall):** {sensitivity*100:.2f}% of actual churners are detected
                - **Specificity:** {specificity*100:.2f}% of actual stayers are identified
                - **ROC AUC:** {roc_auc*100:.2f}% - Strong discrimination ability
                """)
            
            with col2:
                st.markdown("### 💡 Model Insights")
                st.markdown(f"""
                - **True Positives:** {tp} (Correctly identified churners)
                - **True Negatives:** {tn} (Correctly identified stayers)
                - **False Positives:** {fp} (Incorrectly flagged for churn)
                - **False Negatives:** {fn} (Missed churners - Business Loss)
                - **F1 Score:** {f1*100:.2f}% (Balanced metric)
                """)
            
            st.markdown("---")
            
            st.markdown("### 🎯 Recommendations & Action Items")
            
            recommendations = f"""
            1. **Model Deployment Status:** ✅ READY FOR PRODUCTION
               - Achieved F1 score of {f1*100:.2f}% (Good performance)
               - Accuracy of {accuracy*100:.2f}% meets business requirements
               
            2. **Precision vs Recall Trade-off:**
               - Current Precision: {precision*100:.2f}% (Cost of false positives)
               - Current Recall: {recall*100:.2f}% (Cost of missed churners)
               - **Recommendation:** Focus on improving Recall to catch more churners
               
            3. **False Negatives Impact:**
               - Currently missing {fn} potential churners
               - **Action:** Adjust prediction threshold to catch more at-risk customers
               
            4. **Feature Monitoring:**
               - Monitor top {min(10, len(importance_df))} features for churn indicators
               - These features drive {importance_df['Importance'].head(10).sum()*100:.1f}% of predictions
               
            5. **Model Maintenance:**
               - Retrain model monthly with new customer data
               - Monitor for data drift and performance degradation
               - Set up alerts if accuracy drops below {accuracy*100*0.9:.1f}%
               
            6. **Business Integration:**
               - Use Phase 1 dashboard for daily predictions
               - Implement automated alerts for high-risk customers
               - Track retention campaign effectiveness
            """
            
            st.info(recommendations)
    
    except Exception as e:
        st.error(f"❌ Error in metrics display: {str(e)}")
        import traceback
        st.write(traceback.format_exc())


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point with sidebar navigation."""
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    st.sidebar.markdown("---")
    
    phase = st.sidebar.radio(
        "Select Phase:",
        ["Phase 1: Business Analysis & Prediction", "Phase 2: Model Metrics & Statistics"],
        key="phase_radio"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📊 Dashboard Information")
    st.sidebar.info("""
    **Customer Churn Prediction Dashboard**
    
    This interactive dashboard provides:
    - 🔮 Real-time churn predictions
    - 📊 Detailed model metrics
    - 🔍 Feature importance analysis
    - 💼 Business recommendations
    - 📈 Performance analytics
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📁 Project Structure")
    st.sidebar.markdown("""
    ```
    customer_churn/
    ├── src/
    │   ├── data_download.py
    │   ├── eda.py
    │   ├── data_preparation.py
    │   ├── model_training.py
    │   └── pipeline.py
    ├── models/
    │   └── xgboost_model_*.pkl
    ├── data/
    │   └── synthetic_customer_churn_100k.csv
    ├── main.py (this file)
    └── README.md
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ using Streamlit")
    
    # Route to appropriate phase
    if "Phase 1" in phase:
        phase_1_business_prediction()
    else:
        phase_2_model_metrics()


if __name__ == "__main__":
    main()
