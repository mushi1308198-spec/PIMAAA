import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Load model and data
@st.cache_resource
def load_model_and_info():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, scaler, info

model, scaler, model_info = load_model_and_info()

# Sidebar
st.sidebar.title("📊 Diabetes Prediction Model")
st.sidebar.info(
    f"""
    **Best Model:** {model_info['best_model_name']}
    
    **Accuracy:** {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.2f}%)
    
    This model predicts the likelihood of diabetes based on medical measurements.
    """
)

# Main title
st.title("🏥 Pima Indians Diabetes Prediction Model")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose Concentration (mg/dL)", 0, 200, 120)
        diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 0, 122, 70)
        triceps = st.slider("Triceps Skin Fold Thickness (mm)", 0, 99, 20)
    
    with col2:
        serum_insulin = st.slider("2-Hour Serum Insulin (μU/ml)", 0, 846, 100)
        bmi = st.slider("Body Mass Index (kg/m²)", 0.0, 67.1, 25.0)
        diabetes_pedi = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
        age = st.slider("Age (years)", 21, 81, 40)
    
    if st.button("🔍 Predict", use_container_width=True):
        # Prepare input
        input_data = np.array([[pregnancies, glucose, diastolic_bp, triceps, 
                                serum_insulin, bmi, diabetes_pedi, age]])
        
        # Use same preprocessing as training
        if model_info['best_model_name'] == 'Logistic Regression':
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0]
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ HIGH RISK - Diabetes Likely", icon="🔴")
                st.metric("Risk Level", "High", delta="81%")
            else:
                st.success("✅ LOW RISK - No Diabetes", icon="✅")
                st.metric("Risk Level", "Low", delta="19%")
        
        with col2:
            st.metric("Confidence", f"{max(probability)*100:.1f}%")
            st.metric("No Diabetes Probability", f"{probability[0]*100:.1f}%")
            st.metric("Diabetes Probability", f"{probability[1]*100:.1f}%")

with tab2:
    st.header("Model Performance Metrics")
    
    # Display results for all models
    results_data = []
    for model_name, metrics in model_info['all_results'].items():
        results_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    best_metrics = model_info['all_results'][model_info['best_model_name']]
    
    with col1:
        st.metric("Best Accuracy", f"{best_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_metrics['recall']:.4f}")
    with col4:
        st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")

with tab3:
    st.header("About This Model")
    
    st.markdown("""
    ### Dataset: Pima Indians Diabetes Database
    This classic dataset contains medical measurements from the Pima Indians population, 
    focused on predicting diabetes occurrence.
    
    ### Features:
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration
    - **Blood Pressure**: Diastolic blood pressure (mm Hg)
    - **Skin Thickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-hour serum insulin (μU/ml)
    - **BMI**: Body mass index (weight in kg/(height in m)²)
    - **Diabetes Pedigree**: Diabetes pedigree function
    - **Age**: Age in years
    
    ### Models Compared:
    1. **Logistic Regression** - Linear classification model
    2. **Random Forest** - Ensemble of decision trees
    3. **Gradient Boosting** - Sequential tree boosting
    4. **XGBoost** - Extreme Gradient Boosting (optimized implementation)
    
    ### Best Model: 
    **""" + model_info['best_model_name'] + """** with an accuracy of **""" + f"{model_info['accuracy']:.2%}" + """**
    
    ### Disclaimer:
    This model is for educational and informational purposes only. 
    It should not be used for actual medical diagnosis. 
    Please consult with a healthcare professional for medical advice.
    """)
    
    st.info("🔬 Model trained on historical Pima Indians Diabetes Database", icon="ℹ️")
