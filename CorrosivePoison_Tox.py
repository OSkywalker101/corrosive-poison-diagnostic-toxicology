import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Corrosive Poison Diagnostic System",
    page_icon="☠️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e2e8f0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(22, 33, 62, 0.6);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Typography */
.main-header {
    font-size: 3.5rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #ff4b2b, #ff416c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    padding-bottom: 0.5rem;
    text-shadow: 0 4px 10px rgba(255, 65, 108, 0.3);
}
.sub-header {
    font-size: 1.3rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 3rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px 10px 0 0;
    border: 1px solid rgba(255,255,255,0.1);
    border-bottom: none;
    padding: 10px 20px;
    color: #cbd5e1;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(255, 65, 108, 0.1);
    color: #ff416c;
    border-top: 2px solid #ff416c;
}

/* Buttons */
.stButton > button {
    border-radius: 20px;
    background: linear-gradient(45deg, #ff416c, #ff4b2b) !important;
    color: white !important;
    border: none;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6);
}

/* Metrics and Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def generate_toxicology_dataset(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    data = {
        'Oropharyngeal_Burns': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Teeth_Discoloration': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25]),
        'Abdominal_Distension': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'Skin_Lesions': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        'Melena': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Hematemesis': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        '喉部疼痛': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        '吞咽困难': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'Chest_Pain': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
        'Acidosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    acid_type = np.random.choice(['Sulfuric Acid (H2SO4)', 'Nitric Acid (HNO3)', 'Hydrochloric Acid (HCl)'], n_samples)
    
    for i in range(n_samples):
        if acid_type[i] == 'Sulfuric Acid (H2SO4)':
            data['Oropharyngeal_Burns'][i] = np.random.choice([0, 1], p=[0.1, 0.9])
            data['Teeth_Discoloration'][i] = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
            data['Abdominal_Distension'][i] = np.random.choice([0, 1], p=[0.15, 0.85])
            data['Skin_Lesions'][i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            data['Melena'][i] = np.random.choice([0, 1], p=[0.3, 0.7])
            data['Hematemesis'][i] = np.random.choice([0, 1], p=[0.25, 0.75])
            data['Acidosis'][i] = np.random.choice([0, 1], p=[0.2, 0.8])
            
        elif acid_type[i] == 'Nitric Acid (HNO3)':
            data['Oropharyngeal_Burns'][i] = np.random.choice([0, 1], p=[0.2, 0.8])
            data['Teeth_Discoloration'][i] = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            data['Abdominal_Distension'][i] = np.random.choice([0, 1], p=[0.35, 0.65])
            data['Skin_Lesions'][i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            data['Melena'][i] = np.random.choice([0, 1], p=[0.5, 0.5])
            data['Hematemesis'][i] = np.random.choice([0, 1], p=[0.45, 0.55])
            data['Acidosis'][i] = np.random.choice([0, 1], p=[0.5, 0.5])
            
        else:
            data['Oropharyngeal_Burns'][i] = np.random.choice([0, 1], p=[0.35, 0.65])
            data['Teeth_Discoloration'][i] = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            data['Abdominal_Distension'][i] = np.random.choice([0, 1], p=[0.5, 0.5])
            data['Skin_Lesions'][i] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            data['Melena'][i] = np.random.choice([0, 1], p=[0.6, 0.4])
            data['Hematemesis'][i] = np.random.choice([0, 1], p=[0.55, 0.45])
            data['Acidosis'][i] = np.random.choice([0, 1], p=[0.7, 0.3])
    
    df = pd.DataFrame(data)
    df['Acid_Type'] = acid_type
    
    return df

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    results = {}
    
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    return results

def get_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)
        return feat_imp
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_[0])
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': coefs})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)
        return feat_imp
    return None

def predict_acid(model, symptoms, feature_columns):
    input_data = {}
    for col in feature_columns:
        input_data[col] = [symptoms.get(col, 0)]
    
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_df)[0]
        confidence = max(proba) * 100
    else:
        confidence = None
    
    return prediction, confidence

st.markdown('<p class="main-header">☠️ Corrosive Poison Diagnostic System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Toxicology Analysis for Corrosive Agent Identification</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Training", "🔍 Prediction", "📈 Evaluation", "📋 Case Records"])

with st.sidebar:
    st.markdown("### ⚙️ Dataset & Training Config")
    n_samples = st.slider("Samples to generate", 100, 2000, 500)
    test_size = st.slider("Test set ratio", 0.1, 0.4, 0.2)
    random_state = st.number_input("Random seed", 1, 999, 42)
    
    if st.button("🔄 Generate & Train Models", type="primary"):
        with st.spinner("Processing..."):
            df = generate_toxicology_dataset(n_samples=n_samples, random_state=random_state)
            df.to_csv(DATA_DIR / f"toxicology_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            
            feature_cols = [c for c in df.columns if c != 'Acid_Type']
            X = df[feature_cols]
            y = df['Acid_Type']
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['label_encoder'] = label_encoder
            st.session_state['feature_cols'] = feature_cols
            st.session_state['df'] = df
            st.session_state['trained'] = True
            
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, label_encoder
            )
            st.session_state['model_results'] = results
            
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]['model']
            
            joblib.dump(best_model, MODEL_DIR / 'best_toxicology_model.pkl')
            joblib.dump(label_encoder, MODEL_DIR / 'label_encoder.pkl')
            
            st.session_state['best_model_name'] = best_model_name
            st.session_state['best_model'] = best_model
            
        st.success(f"Trained successfully! Best: {best_model_name}")

with tab1:
    st.header("📊 Dataset & Model Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Overview")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write(f"**Total samples:** {len(df)}")
            st.write(f"**Features:** {len(df.columns) - 1}")
            st.write(f"**Classes:** {df['Acid_Type'].nunique()}")
            
            st.write("**Class distribution:**")
            class_dist = df['Acid_Type'].value_counts()
            st.dataframe(class_dist, use_container_width=True)
            
            st.write("**Sample data:**")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("Generate a dataset first to see statistics.")

    if 'model_results' in st.session_state:
        st.subheader("Model Performance Comparison")
        results = st.session_state['model_results']
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': f"{result['accuracy']*100:.2f}%",
                'CV Mean': f"{result['cv_mean']*100:.2f}%",
                'CV Std': f"±{result['cv_std']*100:.2f}%"
            })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

with tab2:
    st.header("Acid Prediction from Symptoms")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train a model first in the Training tab.")
    else:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Clinical Symptoms")
            
            symptoms = {}
            
            symptoms['Oropharyngeal_Burns'] = st.checkbox("Oropharyngeal Burns", help="Burns in the mouth and throat area")
            teeth_options = {0: "None", 1: "Yellow discoloration", 2: "Chalky white discoloration"}
            symptoms['Teeth_Discoloration'] = st.selectbox("Teeth Discoloration", options=[0, 1, 2], format_func=lambda x: teeth_options[x])
            symptoms['Abdominal_Distension'] = st.checkbox("Abdominal Distension", help="Swelling of the abdomen")
            skin_options = {0: "None", 1: "Mild erythema", 2: "Severe burns/necrosis"}
            symptoms['Skin_Lesions'] = st.selectbox("Skin Lesions", options=[0, 1, 2], format_func=lambda x: skin_options[x])
            symptoms['Melena'] = st.checkbox("Melena", help="Black, tarry stools (upper GI bleeding)")
            symptoms['Hematemesis'] = st.checkbox("Hematemesis", help="Vomiting blood")
            symptoms['喉部疼痛'] = st.checkbox("Throat Pain (喉部疼痛)", help="Pain in the throat")
            symptoms['吞咽困难'] = st.checkbox("Dysphagia (吞咽困难)", help="Difficulty swallowing")
            symptoms['Chest_Pain'] = st.checkbox("Chest Pain")
            symptoms['Acidosis'] = st.checkbox("Metabolic Acidosis", help="Detected via blood gas analysis")
        
        with col_right:
            st.subheader("Prediction Result")
            
            if st.button("🔮 Predict Corrosive Agent", type="primary"):
                model = st.session_state['best_model']
                label_encoder = st.session_state['label_encoder']
                feature_cols = st.session_state['feature_cols']
                
                prediction, confidence = predict_acid(model, symptoms, feature_cols)
                
                acid_colors = {
                    'Sulfuric Acid (H2SO4)': '#ff4d4d',
                    'Nitric Acid (HNO3)': '#ffa500',
                    'Hydrochloric Acid (HCl)': '#4d94ff'
                }
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #dc3545 0%, #ff8c00 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                ">
                    <h2 style="margin: 0;">{prediction}</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                proba = model.predict_proba(pd.DataFrame([{k: symptoms.get(k, 0) for k in feature_cols}]))[0]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=label_encoder.classes_,
                    y=proba,
                    marker_color=[acid_colors.get(c, '#808080') for c in label_encoder.classes_],
                    marker_line_width=0,
                    opacity=0.85
                ))
                fig.update_layout(
                    title="Probability by Acid Type",
                    yaxis_title="Probability",
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40),
                    font=dict(family="Outfit", color="#e2e8f0")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state['last_prediction'] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symptoms': symptoms.copy(),
                    'prediction': prediction,
                    'confidence': confidence
                }
        
        if 'last_prediction' in st.session_state:
            with st.expander("📋 View Last Prediction Details"):
                pred = st.session_state['last_prediction']
                st.json({
                    'Timestamp': pred['timestamp'],
                    'Predicted Agent': pred['prediction'],
                    'Confidence': f"{pred['confidence']:.1f}%",
                    'Symptoms': {k: ('Yes' if v == 1 else ('Moderate' if v == 2 else 'No')) if isinstance(v, (int, np.integer)) and k in ['Oropharyngeal_Burns', 'Abdominal_Distension', 'Melena', 'Hematemesis', '喉部疼痛', '吞咽困难', 'Chest_Pain', 'Acidosis'] else v for k, v in pred['symptoms'].items()}
                })

with tab3:
    st.header("Model Evaluation")
    
    if 'model_results' not in st.session_state:
        st.warning("Please train models first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Classification Report")
            model_name = st.selectbox("Select Model", options=list(st.session_state['model_results'].keys()))
            
            model = st.session_state['model_results'][model_name]['model']
            y_pred = st.session_state['model_results'][model_name]['predictions']
            y_test = st.session_state['y_test']
            label_encoder = st.session_state['label_encoder']
            
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                x=label_encoder.classes_,
                y=label_encoder.classes_,
                color_continuous_scale='Sunsetdark',
                text_auto=True,
                title="Confusion Matrix"
            )
            fig.update_layout(
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit", color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance")
        
        feat_imp = get_feature_importance(model, st.session_state['feature_cols'], model_name)
        if feat_imp is not None:
            fig = px.bar(
                feat_imp.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='Sunset'
            )
            fig.update_layout(
                height=400, 
                yaxis={'categoryorder': 'total ascending'},
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit", color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {model_name}")

with tab4:
    st.header("Case Records")
    
    st.info("📋 Case logging functionality - save predictions for later review")
    
    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        
        case_record = {
            'case_id': f"CASE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': pred['timestamp'],
            'predicted_agent': pred['prediction'],
            'confidence': pred['confidence'],
            'symptoms': pred['symptoms']
        }
        
        st.json(case_record)
        
        case_df = pd.DataFrame([case_record])
        
        csv = case_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Case Record",
            data=csv,
            file_name=f"case_record_{case_record['case_id']}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("☠️ Corrosive Poison Diagnostic System | For clinical decision support only")