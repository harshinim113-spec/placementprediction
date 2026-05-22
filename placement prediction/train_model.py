"""
train_model.py - Model Training Script for Placement Prediction System
======================================================================
Trains a Random Forest Classifier on the student dataset,
evaluates performance, and saves the trained model and scaler to disk.
Run this script once before launching the application.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train():
    """
    Complete training pipeline:
    1. Load dataset
    2. Preprocess data
    3. Split into train/test sets
    4. Scale features
    5. Train Random Forest model
    6. Evaluate performance
    7. Save model and scaler
    """
    print("=" * 60)
    print("   PLACEMENT PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load the dataset ──────────────────────────────
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    
    if not os.path.exists(dataset_path):
        print("\n❌ Error: dataset.csv not found!")
        print(f"   Expected at: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total records: {len(df)}")
    print(f"   Features: {list(df.columns)}")

    # ── Step 2: Data Preprocessing ────────────────────────────
    print("\n📊 Data Overview:")
    print(f"   Placed: {df['Placed'].sum()} students")
    print(f"   Not Placed: {(df['Placed'] == 0).sum()} students")
    print(f"   Placement Rate: {df['Placed'].mean() * 100:.1f}%")

    # Separate features and target
    feature_columns = [
        'CGPA', 'Aptitude_Score', 'Communication_Skills',
        'Technical_Skills', 'Internship_Count', 'Project_Count',
        'Attendance_Percentage', 'Certifications_Count'
    ]
    
    X = df[feature_columns].values  # Feature matrix
    y = df['Placed'].values          # Target variable

    # Check for missing values
    if df[feature_columns].isnull().any().any():
        print("\n⚠️  Missing values detected! Filling with column means...")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        X = df[feature_columns].values

    # ── Step 3: Train-Test Split ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📂 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")

    # ── Step 4: Feature Scaling ───────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n⚙️  Features scaled using StandardScaler")

    # ── Step 5: Train Random Forest Classifier ────────────────
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,       # Number of trees in the forest
        max_depth=10,           # Maximum depth of each tree
        min_samples_split=5,    # Minimum samples to split a node
        min_samples_leaf=2,     # Minimum samples at each leaf node
        random_state=42,        # For reproducibility
        n_jobs=-1               # Use all CPU cores
    )
    model.fit(X_train_scaled, y_train)
    print("   ✅ Model training complete!")

    # ── Step 6: Evaluate Performance ──────────────────────────
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📈 Model Performance:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:")
    print(f"   {cm[0]}")
    print(f"   {cm[1]}")

    # Feature Importances
    feature_names = [
        "CGPA", "Aptitude", "Communication", "Technical",
        "Internships", "Projects", "Attendance", "Certifications"
    ]
    importances = model.feature_importances_
    print(f"\n🔍 Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"   {name:15s} {imp:.4f} {bar}")

    # ── Step 7: Save Model and Scaler ─────────────────────────
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(model_dir, "placement_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {scaler_path}")

    print("\n" + "=" * 60)
    print("   ✅ TRAINING COMPLETE - Model is ready for predictions!")
    print("=" * 60)

    return accuracy


if __name__ == "__main__":
    train()
