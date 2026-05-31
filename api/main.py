from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import json
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Corrosive Poison Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

class TrainingConfig(BaseModel):
    n_samples: int = 500
    test_size: float = 0.2
    random_state: int = 42

class SymptomInput(BaseModel):
    Oropharyngeal_Burns: int = 0
    Teeth_Discoloration: int = 0
    Abdominal_Distension: int = 0
    Skin_Lesions: int = 0
    Melena: int = 0
    Hematemesis: int = 0
    throat_pain: int = 0
    dysphagia: int = 0
    Chest_Pain: int = 0
    Acidosis: int = 0

state = {
    "trained": False,
    "best_model": None,
    "label_encoder": None,
    "feature_cols": None,
    "model_results": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "df": None
}

def generate_toxicology_dataset(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    data = {
        'Oropharyngeal_Burns': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Teeth_Discoloration': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25]),
        'Abdominal_Distension': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'Skin_Lesions': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        'Melena': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Hematemesis': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'throat_pain': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'dysphagia': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
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

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
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
            'accuracy': float(accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'predictions': y_pred.tolist()
        }
    
    return results

@app.get("/")
def root():
    return {"message": "Corrosive Poison Diagnostic API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "trained": state["trained"]}

@app.post("/api/train")
def train(config: TrainingConfig):
    df = generate_toxicology_dataset(n_samples=config.n_samples, random_state=config.random_state)
    df.to_csv(DATA_DIR / f"toxicology_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    feature_cols = [c for c in df.columns if c != 'Acid_Type']
    X = df[feature_cols]
    y = df['Acid_Type']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=config.test_size, random_state=config.random_state, stratify=y_encoded
    )
    
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    
    state["trained"] = True
    state["df"] = df
    state["feature_cols"] = feature_cols
    state["label_encoder"] = label_encoder
    state["model_results"] = results
    state["best_model_name"] = best_model_name
    
    return {
        "status": "success",
        "message": f"Models trained successfully! Best model: {best_model_name}",
        "best_model": best_model_name,
        "best_accuracy": results[best_model_name]['accuracy'],
        "model_results": results,
        "dataset_info": {
            "total_samples": len(df),
            "features": len(feature_cols),
            "classes": df['Acid_Type'].nunique(),
            "class_distribution": df['Acid_Type'].value_counts().to_dict()
        }
    }

@app.post("/api/predict")
def predict(symptoms: SymptomInput):
    if not state["trained"]:
        return {"error": "Model not trained. Please train first."}
    
    model_names = {
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    feature_cols = state["feature_cols"]
    label_encoder = state["label_encoder"]
    
    input_data = {
        'Oropharyngeal_Burns': symptoms.Oropharyngeal_Burns,
        'Teeth_Discoloration': symptoms.Teeth_Discoloration,
        'Abdominal_Distension': symptoms.Abdominal_Distension,
        'Skin_Lesions': symptoms.Skin_Lesions,
        'Melena': symptoms.Melena,
        'Hematemesis': symptoms.Hematemesis,
        'throat_pain': symptoms.throat_pain,
        'dysphagia': symptoms.dysphagia,
        'Chest_Pain': symptoms.Chest_Pain,
        'Acidosis': symptoms.Acidosis
    }
    
    input_df = pd.DataFrame([input_data])
    
    model = model_names[state["best_model_name"]]
    
    X = state["df"][[c for c in state["df"].columns if c != 'Acid_Type']]
    y = state["df"]['Acid_Type']
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    model.fit(X_train, y_train)
    
    prediction = model.predict(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    
    proba = model.predict_proba(input_df)[0]
    confidence = float(max(proba)) * 100
    
    acid_colors = {
        'Sulfuric Acid (H2SO4)': '#ff4d4d',
        'Nitric Acid (HNO3)': '#ffa500',
        'Hydrochloric Acid (HCl)': '#4d94ff'
    }
    
    probabilities = {
        label: float(proba[i]) 
        for i, label in enumerate(label_encoder.classes_)
    }
    
    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "symptoms": input_data,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/api/evaluation")
def evaluation():
    if not state["trained"]:
        return {"error": "Model not trained. Please train first."}
    
    model_names = {
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    label_encoder = state["label_encoder"]
    X = state["df"][[c for c in state["df"].columns if c != 'Acid_Type']]
    y = state["df"]['Acid_Type']
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    evaluation_results = {}
    
    for name, model in model_names.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation_results[name] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "accuracy": float(report['accuracy'])
        }
    
    return {
        "models": evaluation_results,
        "classes": label_encoder.classes_.tolist()
    }

@app.get("/api/overview")
def overview():
    if not state["trained"]:
        return {"error": "Model not trained. Please train first."}
    
    df = state["df"]
    results = state["model_results"]
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'model': name,
            'test_accuracy': f"{result['accuracy']*100:.2f}%",
            'cv_mean': f"{result['cv_mean']*100:.2f}%",
            'cv_std': f"±{result['cv_std']*100:.2f}%"
        })
    
    return {
        "dataset": {
            "total_samples": len(df),
            "features": len(df.columns) - 1,
            "classes": df['Acid_Type'].nunique(),
            "class_distribution": {k: int(v) for k, v in df['Acid_Type'].value_counts().to_dict().items()},
            "sample_data": df.head(10).to_dict(orient='records')
        },
        "model_comparison": comparison_data,
        "best_model": state["best_model_name"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)