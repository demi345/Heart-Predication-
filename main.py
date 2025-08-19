import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model & Prediction Functions ---
# Define these at the top to avoid NameError
algonames = ['Decision Tree Classifier', 'SVC', 'Random Forest Classifier']
modelnames = ['DecisionTreeClassifier.pkl', 'SVC.pkl', 'RandomForestClassifier.pkl']

@st.cache_data
def predict_heart_disease(data):
    """Loads models and returns predictions with probabilities."""
    predictions = []
    for modelname in modelnames:
        try:
            with open(modelname, 'rb') as file:
                model = pickle.load(file)
                # Handle SVC which may not have predict_proba
                if modelname == 'SVC.pkl' and not hasattr(model, 'predict_proba'):
                    # Use predict() and create a placeholder for probability
                    prediction = model.predict(data)
                    # Create a structure that mimics predict_proba output
                    proba_output = np.zeros((1, 2))
                    proba_output[0, prediction[0]] = 1.0
                    predictions.append(proba_output)
                else:
                    prediction_proba = model.predict_proba(data)
                    predictions.append(prediction_proba)
        except FileNotFoundError:
            st.error(f"Model file not found: {modelname}. Please ensure it's in the same folder as the app.")
            return None
        except Exception as e:
            st.error(f"An error occurred loading model {modelname}: {e}")
            return None
    return predictions

def preprocess_data(df):
    """Preprocesses the input DataFrame for prediction."""
    # Map categorical features
    sex_map = {"Male": "M", "Female": "F"}
    chest_pain_map = {"Typical Angina": "TA", "Atypical Angina": "ATA", "Non-Anginal Pain": "NAP", "Asymptomatic": "ASY"}
    fasting_bs_map = {True: 1, False: 0, "True": 1, "False": 0}
    restecg_map = {"Normal": "Normal", "ST-T Wave Abnormality": "ST", "Left Ventricular Hypertrophy": "LVH"}
    exercise_angina_map = {"Yes": "Y", "No": "N"}
    slope_map = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

    # Apply mappings - handle potential string/bool issues in FastingBS
    df['Sex'] = df['Sex'].map(sex_map)
    df['ChestPainType'] = df['ChestPainType'].map(chest_pain_map)
    df['FastingBS'] = df['FastingBS'].map(fasting_bs_map)
    df['RestingECG'] = df['RestingECG'].map(restecg_map)
    df['ExerciseAngina'] = df['ExerciseAngina'].map(exercise_angina_map)
    df['ST_Slope'] = df['ST_Slope'].map(slope_map)

    # Feature Engineering
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 55, 70, 120], labels=['<40', '40-55', '55-70', '70+'], right=False)
    df['Age_Cholesterol'] = df['Age'] * df['Cholesterol']

    # One-Hot Encoding
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'AgeGroup']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Align columns with training data
    training_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Age_Cholesterol', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up', 'AgeGroup_40-55', 'AgeGroup_55-70', 'AgeGroup_70+']
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[training_columns]

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This app uses machine learning to predict the likelihood of heart disease. "
        "Enter patient data on the 'Single Prediction' tab or upload a file for bulk prediction."
    )
    st.success("Navigate through the tabs to explore different features.")
    st.markdown("The dataset for this project was sourced from Kaggle.")
    st.markdown("---")
    st.markdown("Created by **Demiana Abadir**")

# Header
col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.title("Heart Disease Prediction Dashboard")
    st.markdown("This application uses several machine learning models to predict the likelihood of heart disease based on patient data.")
with col2:
    # The heart.png image is removed to prevent errors when the file is not found.
    pass

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Prediction", "Bulk Prediction", "Exploratory Data Analysis", "Model Info", "Dataset Info"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.header("Enter Patient Data")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", options=["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])
        restecg = st.selectbox("Resting ECG", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])

    if st.button("Get Prediction"):
        # --- Preprocessing ---
        input_data_dict = {'Age': age, 'Sex': sex, 'ChestPainType': chest_pain, 'RestingBP': trestbps, 'Cholesterol': chol, 'FastingBS': fasting_bs, 'RestingECG': restecg, 'MaxHR': thalach, 'ExerciseAngina': exercise_angina, 'ST_Slope': slope}
        input_df = pd.DataFrame([input_data_dict])

        # Use the preprocessing function
        input_data = preprocess_data(input_df)

        # --- Prediction ---
        st.subheader("Prediction Results")
        result_probas = predict_heart_disease(input_data)
        if result_probas:
            for i in range(len(result_probas)):
                prediction = np.argmax(result_probas[i])
                # Check if confidence is meaningful (not 0 or 100 from our placeholder)
                is_svc_without_proba = (modelnames[i] == 'SVC.pkl' and (result_probas[i][0][prediction] == 1.0 or result_probas[i][0][prediction] == 0.0))

                st.markdown(f"**{algonames[i]}**")

                if is_svc_without_proba:
                    if prediction == 1:
                        st.error("Heart Disease Detected 💔 (Confidence score not available for this model)")
                    else:
                        st.success("No Heart Disease Detected ✅ (Confidence score not available for this model)")
                else:
                    confidence = result_probas[i][0][prediction] * 100
                    if prediction == 1:
                        st.error(f"Heart Disease Detected 💔 (Confidence: {confidence:.2f}%)")
                    else:
                        st.success(f"No Heart Disease Detected ✅ (Confidence: {confidence:.2f}%)")
                st.markdown("---")

# --- Tab 2: Bulk Prediction ---
with tab2:
    st.header("Bulk Heart Disease Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with patient data", type="csv")
    if uploaded_file:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", bulk_df.head())

            # Preprocess the entire dataframe
            processed_bulk_df = preprocess_data(bulk_df.copy())

            # Get predictions for the processed data
            all_models_predictions = []
            for model_name in modelnames:
                try:
                    with open(model_name, 'rb') as file:
                        model = pickle.load(file)
                        if hasattr(model, 'predict'):
                            preds = model.predict(processed_bulk_df)
                            all_models_predictions.append(preds)
                        else:
                            st.error(f"Model {model_name} does not have a 'predict' method.")
                            all_models_predictions.append(np.array(['Error'] * len(processed_bulk_df)))
                except pickle.UnpicklingError:
                    st.error(f"Error loading '{model_name}'. This file is not a valid pickle file. Please check or regenerate it.")
                    # Add placeholder predictions to avoid crashing the app
                    all_models_predictions.append(np.array(['Error'] * len(processed_bulk_df)))
                    continue

            # Create a results dataframe
            results_df = bulk_df.copy()
            for i, name in enumerate(algonames):
                results_df[f'{name} Prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in all_models_predictions[i]]

            st.subheader("Prediction Results for Uploaded File")
            st.write(results_df)

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_output = convert_df_to_csv(results_df)

            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name='bulk_predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.warning("Please ensure the CSV file has the correct columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, ST_Slope.")


# --- Tab 3: Exploratory Data Analysis ---
with tab3:
    st.header("Exploratory Data Analysis")
    st.markdown("Static analysis of the dataset features.")

    try:
        df_eda = pd.read_csv(r"C:\Users\dabadir\Desktop\.py\notebooks\datasets\heart.csv")

        # --- Age Distribution ---
        st.subheader("Age Distribution")
        fig_age, ax_age = plt.subplots(figsize=(6, 4))
        sns.histplot(df_eda['Age'], kde=True, ax=ax_age, color='skyblue')

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig_age)

        # --- Heart Disease Count ---
        st.subheader("Heart Disease Vs No Heart Disease")
        fig_count, ax_count = plt.subplots(figsize=(6, 4))
        sns.countplot(x='HeartDisease', data=df_eda, ax=ax_count, palette='pastel')
        ax_count.set_xticklabels(['No Heart Disease', 'Heart Disease'])

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig_count)

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap of Numerical Features")
        numerical_df = df_eda.select_dtypes(include=np.number)
        corr = numerical_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    except FileNotFoundError:
        st.error("Dataset file ('heart.csv') not found for EDA.")
    except Exception as e:
        st.error(f"An error occurred during EDA: {e}")


# --- Tab 4: Model Info ---
with tab4:
    st.header("Model Performance")
    # This data is static and not linked to your notebook's output.
    data = {"Decision Tree Classifier":  0.788043, "SVC": 0.717391, "RandomForestClassifier":0.869565 }
    df_acc = pd.DataFrame(list(data.items()), columns=['Model', 'Accuracy'])

    # Define custom colors for the models
    color_map = {
        "Decision Tree Classifier": "salmon",
        "SVC": "lightblue",
        "RandomForestClassifier": "lightgreen"
    }

    fig = px.bar(
        df_acc,
        y='Accuracy',
        x='Model',
        title="Model Accuracies",
        color='Model',
        text_auto='.3f',
        color_discrete_map=color_map  # Apply the custom color map
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Dataset Info ---
with tab5:
    st.header("About the Dataset")
    st.markdown("""
    This application uses the **Heart Failure Prediction Dataset**. This dataset combines five major heart disease datasets and contains 11 common features.

    **Source:** [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    """)

    st.subheader("Dataset Preview")
    try:
        df_info = pd.read_csv(r"C:\Users\dabadir\Desktop\.py\notebooks\datasets\heart.csv")
        st.dataframe(df_info.head())
        st.subheader("Dataset Columns and Data Types")
        from io import StringIO
        buffer = StringIO()
        df_info.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.subheader("Numerical Summary")
        st.write(df_info.describe())
    except FileNotFoundError:
        st.error("Dataset file ('heart.csv') not found. Cannot display dataset information.")
    except Exception as e:
        st.error(f"An error occurred during dataset info: {e}")
