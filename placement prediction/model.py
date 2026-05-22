"""
model.py - Machine Learning Model Module for Placement Prediction
=================================================================
Loads the trained Random Forest model and scaler, provides prediction
functionality with probability scores and personalized improvement suggestions.
"""

import os
import pickle
import numpy as np

# Paths to the saved model and scaler
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def load_model():
    """
    Load the trained Random Forest model and StandardScaler from disk.
    Returns (model, scaler) tuple or (None, None) if files not found.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_placement(cgpa, aptitude, communication, technical,
                      internships, projects, attendance, certifications):
    """
    Predict placement status for a student based on their profile.

    Parameters:
        cgpa (float): Cumulative GPA (0-10)
        aptitude (float): Aptitude test score (0-100)
        communication (int): Communication skills rating (1-10)
        technical (int): Technical skills rating (1-10)
        internships (int): Number of internships completed
        projects (int): Number of projects completed
        attendance (float): Attendance percentage (0-100)
        certifications (int): Number of certifications

    Returns:
        dict: Contains 'prediction', 'probability', and 'suggestions'
    """
    model, scaler = load_model()
    if model is None:
        return {
            "prediction": "Error",
            "probability": 0.0,
            "suggestions": ["Model not found. Please train the model first by running train_model.py"]
        }

    # Prepare input features as a 2D array for the model
    features = np.array([[cgpa, aptitude, communication, technical,
                          internships, projects, attendance, certifications]])

    # Scale features using the same scaler used during training
    features_scaled = scaler.transform(features)

    # Get prediction (0 = Not Placed, 1 = Placed)
    prediction = model.predict(features_scaled)[0]

    # Get probability scores for each class
    probability = model.predict_proba(features_scaled)[0]
    placement_prob = round(probability[1] * 100, 1)  # Probability of being placed

    # Generate personalized suggestions based on weak areas
    suggestions = generate_suggestions(
        cgpa, aptitude, communication, technical,
        internships, projects, attendance, certifications,
        placement_prob
    )

    return {
        "prediction": "Placed" if prediction == 1 else "Not Placed",
        "probability": placement_prob,
        "suggestions": suggestions
    }


def generate_suggestions(cgpa, aptitude, communication, technical,
                         internships, projects, attendance, certifications,
                         probability):
    """
    Generate personalized improvement suggestions based on weak areas.
    Analyzes each parameter against ideal thresholds and provides
    actionable advice for areas that need improvement.
    """
    suggestions = []

    # Analyze each parameter and suggest improvements where needed
    if cgpa < 7.0:
        suggestions.append("📚 Improve your CGPA — aim for 7.5+ by focusing on core subjects and regular revision.")
    elif cgpa < 8.0:
        suggestions.append("📚 Your CGPA is decent. Push it above 8.0 for better placement opportunities.")

    if aptitude < 65:
        suggestions.append("🧠 Practice aptitude questions daily — target platforms like IndiaBix, PrepInsta, or GeeksforGeeks.")
    elif aptitude < 80:
        suggestions.append("🧠 Good aptitude score! Practice more to break the 80+ mark for top companies.")

    if communication < 7:
        suggestions.append("🗣️ Enhance communication skills — join a speaking club, practice mock interviews, and read daily.")
    
    if technical < 7:
        suggestions.append("💻 Strengthen technical skills — learn DSA, practice on LeetCode/HackerRank, and build projects.")
    elif technical < 9:
        suggestions.append("💻 Good technical base! Dive deeper into system design and advanced algorithms.")

    if internships < 1:
        suggestions.append("🏢 Complete at least 1-2 internships — apply on LinkedIn, Internshala, or company career pages.")
    elif internships < 2:
        suggestions.append("🏢 Great start with internships! Try to get one more in a reputed company for stronger profile.")

    if projects < 2:
        suggestions.append("🔧 Build at least 2-3 real-world projects — showcase them on GitHub with proper documentation.")
    elif projects < 4:
        suggestions.append("🔧 Add more diverse projects — include ML, web, or mobile apps to stand out.")

    if attendance < 75:
        suggestions.append("📅 Improve attendance to 75%+ — consistent class attendance helps in exam preparation.")
    elif attendance < 85:
        suggestions.append("📅 Good attendance! Maintain 85%+ for a disciplined academic record.")

    if certifications < 2:
        suggestions.append("🎓 Get certified — pursue courses on Coursera, Udemy, or NPTEL in your domain.")
    elif certifications < 4:
        suggestions.append("🎓 Nice certifications! Add industry-recognized ones like AWS, Google, or Microsoft certs.")

    # Overall assessment
    if probability >= 85:
        suggestions.insert(0, "🌟 Excellent profile! You have a very strong chance of placement. Keep maintaining your standards.")
    elif probability >= 60:
        suggestions.insert(0, "✅ Good profile overall! Focus on the areas mentioned below to boost your chances further.")
    elif probability >= 40:
        suggestions.insert(0, "⚠️ Moderate profile. Significant improvements needed in the areas below for better placement prospects.")
    else:
        suggestions.insert(0, "🔴 Your profile needs substantial improvement. Focus intensively on the suggestions below.")

    return suggestions


def get_feature_importance():
    """
    Get feature importance scores from the trained model.
    Useful for visualizing which factors matter most for placement.
    """
    model, _ = load_model()
    if model is None:
        return None
    
    feature_names = [
        "CGPA", "Aptitude", "Communication", "Technical",
        "Internships", "Projects", "Attendance", "Certifications"
    ]
    importances = model.feature_importances_
    return dict(zip(feature_names, [round(x * 100, 1) for x in importances]))
