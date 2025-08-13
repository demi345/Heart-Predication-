# Heart Disease Prediction Project

## Project Overview
This project implements machine learning models to predict heart disease using various clinical and diagnostic features. Through comprehensive data analysis and machine learning techniques, we aim to accurately identify potential heart disease cases based on patient data.

## Features
- **Data Analysis & Visualization**: Comprehensive exploratory data analysis of heart disease factors
- **Feature Engineering**: Creation of derived features and preprocessing of data
- **Multiple ML Models**:
  - Random Forest Classifier
  - Support Vector Machine (SVC)
  - Decision Tree Classifier
- **Model Evaluation**: Comparison using multiple metrics (Accuracy, Recall, F1-Score)

## Dataset Features
The dataset includes the following clinical parameters:
- **Age**: Patient's age in years
- **Sex**: Biological sex (M/F)
- **ChestPainType**: 
  - TA: Typical Angina
  - NAP: Non-Anginal Pain
  - ASY: Asymptomatic
  - ATA: Atypical Angina
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol level (mg/dl)
- **FastingBS**: Fasting blood sugar
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina
- **ST_Slope**: Slope of ST segment during peak exercise

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
```

## Project Structure
```
.py/
├── datasets/
│   └── heart.csv
├── notebooks/
│   ├── Heart Predication.ipynb
│   ├── DecisionTreeClassifier.pkl
│   ├── RandomForestClassifier.pkl
│   └── SVC.pkl
├── main.py
└── README.md
```

## Key Findings
- RandomForest classifier achieved the highest performance among all models
- Feature importance analysis revealed key predictors of heart disease
- Multiple experiments were conducted with different feature sets to optimize model performance

## Model Performance
Three experiments were conducted with different feature sets:
1. All Features Model
2. Selected Features (Age + Cholesterol + ChestPainType)
3. Extended Feature Set (Age + Cholesterol + ChestPainType + MaxHR + ExerciseAngina + ST_slope)

The RandomForest Classifier consistently performed well across all experiments.
