from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# --- Step 1: Dataset ---
data = {
    "Oropharyngeal_burns": [1, 1, 1],
    "Teeth_change": ["chalky_white", "yellow", "none"],
    "Abdominal_distension": [0, 1, 1],
    "Skin_discoloration": ["black", "yellow", "none"],
    "Perforation": ["common", "less_common", "less_common"],
    "Acid": ["H2SO4", "HNO3", "HCl"]
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = df.copy()
for col in ["Teeth_change", "Skin_discoloration", "Perforation"]:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

X = df_encoded.drop("Acid", axis=1)
y = df_encoded["Acid"]

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# --- Step 2: CLI User Input ---
print("\n--- Corrosive Poison Diagnostic Tool ---")
burns = input("Oropharyngeal burns present? (yes/no): ").lower() == "yes"
teeth = input("Teeth appearance (chalky_white/yellow/none): ").lower()
distension = input("Abdominal distension present? (yes/no): ").lower() == "yes"
skin = input("Skin discoloration (black/yellow/none): ").lower()
perforation = input("Perforation of stomach (common/less_common): ").lower()

# Convert input to model format
input_df = pd.DataFrame({
    "Oropharyngeal_burns": [1 if burns else 0],
    "Teeth_change": [pd.Categorical([teeth], categories=df["Teeth_change"].unique()).codes[0]],
    "Abdominal_distension": [1 if distension else 0],
    "Skin_discoloration": [pd.Categorical([skin], categories=df["Skin_discoloration"].unique()).codes[0]],
    "Perforation": [pd.Categorical([perforation], categories=df["Perforation"].unique()).codes[0]]
})

# Prediction
predicted = model.predict(input_df)[0]
print(f"\n>> Predicted corrosive agent: {predicted}")
