# Corrosive Poison Diagnostic System

An **AI-powered toxicology decision support system** that predicts the most likely corrosive agent (Sulfuric Acid H₂SO₄, Nitric Acid HNO₃, or Hydrochloric Acid HCl) based on observed clinical symptoms.

## Features

- **Interactive Streamlit Interface** - Easy symptom input and prediction
- **Multiple ML Models** - Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Model Evaluation** - Classification reports, confusion matrices, cross-validation
- **Feature Importance Analysis** - Understand which symptoms drive predictions
- **Probability Outputs** - Confidence scores for each prediction
- **Case Logging** - Save and download prediction records

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Models
1. Navigate to the **Training** tab
2. Configure dataset size and test ratio
3. Click **Generate Dataset & Train**
4. Models will be trained and the best model will be auto-selected

### Making Predictions
1. Navigate to the **Prediction** tab
2. Select clinical symptoms from the checkboxes
3. Click **Predict Corrosive Agent**
4. View the predicted agent with confidence score

### Evaluating Models
1. Navigate to the **Evaluation** tab
2. Select a model to view detailed metrics
3. View classification reports and confusion matrices
4. Analyze feature importance

## Dataset

The synthetic dataset includes the following clinical features:
- Oropharyngeal Burns
- Teeth Discoloration
- Abdominal Distension
- Skin Lesions
- Melena (black stool)
- Hematemesis (vomiting blood)
- Throat Pain
- Dysphagia (difficulty swallowing)
- Chest Pain
- Metabolic Acidosis

## Models

| Model | Description |
|-------|-------------|
| Decision Tree | Interpretable tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble method |
| Logistic Regression | Linear probabilistic classifier |
| SVM | Support Vector Machine with RBF kernel |

## Disclaimer

This system is for **clinical decision support only**. All predictions must be verified by a qualified medical professional. The AI model is a tool to assist, not replace, clinical judgment.

## Dataset Source

The dataset was adapted from *[Review of Forensic Medicine and Toxicology by Gautam Biswas]* and expanded with synthetic data for demonstration purposes.