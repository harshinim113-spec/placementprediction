"""
main.py - Entry Point for AI Placement Prediction System
=========================================================
Trains the model if needed, then launches the GUI.
"""

import os
import sys

def main():
    # Train model if not already trained
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement_model.pkl")
    if not os.path.exists(model_path):
        print("First run detected — training model...")
        from train_model import train
        train()
        print()

    # Launch GUI
    from gui import PlacementApp
    app = PlacementApp()
    app.mainloop()

if __name__ == "__main__":
    main()
