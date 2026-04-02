# Crop Recommendation System: A Comparative ML Study

This repository contains a precision agriculture project developed for the Machine Learning course at International Islamic University Malaysia (IIUM). The system predicts the most suitable crops for cultivation based on eight critical environmental and soil parameters.

# Project Overview

Farmers often face risks due to unpredictable soil and weather conditions. We developed an AI-powered solution that achieves 99.32% accuracy in recommending crops, helping to minimize financial loss and promote sustainable farming.

# Key Features
1. Advanced Gradient Boosting: Implementation of CatBoost, LightGBM, and XGBoost algorithms.
2. Hyperparameter Optimization: Used RandomizedSearchCV with 3-fold cross-validation to fine-tune model performance.
3. Feature Importance Analysis: Identified Rainfall, Humidity, and Potassium (K) as the primary drivers for crop suitability.
4. Interactive Deployment: A functional web dashboard built with Streamlit for real-time recommendations.

# Repository Structure
The project is split into baseline and optimized scripts to demonstrate the improvement gained through tuning:
- catboost_baseline.py = Initial implementation of the CatBoost classifier.
- lightgbm_baseline.py = Baseline LightGBM model optimized for training speed.
- xgboost_baseline.py = Standard XGBoost implementation for structured data.
- hypertune_catboost.py = Optimized script reaching 99.32% accuracy.
- hypertune_lightgbm.py = Hyperparameter tuning for LightGBM parameters.
- hypertune_xgboost.py = Tuning script improving XGBoost to 99.09% accuracy.
- app.py = The Streamlit application code for model deployment.

# Technical Stack
- Language: Python 
- Libraries: CatBoost, LightGBM, XGBoost, Scikit-Learn, Pandas, NumPy 
- Deployment: Streamlit 
- Tools: VS Code, Jupyter Notebooks 

# Dataset Information
The models were trained using the Crop Recommendation Dataset, consisting of 2,200 records with the following features:
- Soil Nutrients: Nitrogen (N), Phosphorous (P), Potassium (K).
- Climate Factors: Temperature, Humidity, Rainfall.
- Soil Chemistry: pH levels.

# The Team
- Muhammad Arif Danial (me) – Developed and configured all code; conducted results analysis and discussion.
- Syazwan Fariz – Handled Abstract, Introduction, Deployment Architecture, and Conclusion.
- Syed Muhammad Afiq – Designed Methodology and Experimental Setup.

# Acknowledgments
Special thanks to our instructor, Dr. Amelia Ritahani Bt. Ismail, for her guidance throughout this semester.
