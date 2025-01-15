import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Autism Traits Prediction", layout="wide")

# Apply background color using custom CSS
# Apply background color and border using custom CSS
page_style = """
<style>
body {
    background-color: #fffacd; /* Light yellow color */
    border: 5px solid #000; /* Black border of 5px */
    margin: 10px; /* Add some spacing around the border */
    padding: 10px; /* Add spacing inside the border */
    border-radius: 15px; /* Rounded corners for the border */
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# Title and Description
st.title("Autism Traits Prediction")
st.write("This application predict autism traits in toddlers")

# Add a sidebar for additional options
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this sidebar to upload your dataset and make predictions.")

# Step 1: File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load and display the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(data)

    # Step 3: Preprocess the dataset
    df = data.copy()
    df = df.rename(columns={'Class/ASD Traits ': 'ASD_Traits'})
    df['ASD_Traits'] = df['ASD_Traits'].apply(lambda x: 1 if x.strip() == 'Yes' else 0)
    df = df.drop(columns=['Case_No', 'Who completed the test'], errors='ignore')

    # Encode categorical columns
    categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Define features (X) and target (y)
    if 'ASD_Traits' in df.columns:
        X = df.drop(columns=['ASD_Traits'])
        y = df['ASD_Traits']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 4: Train the model
        model = RandomForestClassifier(
            n_estimators=5,
            max_depth=3,
            max_features=3,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Step 5: Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(f"Model Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Step 6: Visualizations
        st.subheader("Feature Importances:")
        importances = model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(X.shape[1]), importances[indices], align="center")
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels(feature_names[indices], rotation=90)
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        st.pyplot(fig)

        # Pie Chart for Distribution of Target Variable
        st.subheader("Target Variable Distribution")
        target_counts = y.value_counts()
        fig, ax = plt.subplots()
        ax.pie(target_counts, labels=["No ASD Traits", "ASD Traits"], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        st.pyplot(fig)

        # Step 7: User Input for Prediction
        st.sidebar.header("Input Features for Prediction")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        # Display prediction results
        st.subheader("Prediction for Input Data:")
        result = "ASD Traits Detected" if prediction[0] == 1 else "No ASD Traits Detected"
        st.success(result)

        # Check prediction accuracy and data quality
        st.subheader("Analysis of Results:")
        if accuracy > 0.8:
            st.write("The model is performing well with satisfactory accuracy.")
        else:
            st.warning("The model accuracy is low. Consider improving the dataset or model.")

    else:
        st.error("The uploaded dataset does not have the target column 'Class/ASD Traits '.")
else:
    st.info("Please upload a dataset to proceed.")

# import streamlit as st

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt

# # Title of the app
# st.title("Autism Traits Prediction")
# st.write("This app predicts ASD traits in toddlers using a Random Forest model!")

# # Step 1: File upload
# uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# if uploaded_file is not None:
#     # Step 2: Load and display the dataset
#     data = pd.read_csv(uploaded_file)
#     st.write("Uploaded Dataset:")
#     st.dataframe(data)

#     # Step 3: Preprocess the dataset
#     df = data.copy()
#     df = df.rename(columns={'Class/ASD Traits ': 'ASD_Traits'})
#     df['ASD_Traits'] = df['ASD_Traits'].apply(lambda x: 1 if x.strip() == 'Yes' else 0)
#     df = df.drop(columns=['Case_No', 'Who completed the test'], errors='ignore')

#     # Encode categorical columns
#     categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
#     label_encoders = {}
#     for col in categorical_cols:
#         if col in df.columns:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             label_encoders[col] = le

#     # Define features (X) and target (y)
#     if 'ASD_Traits' in df.columns:
#         X = df.drop(columns=['ASD_Traits'])
#         y = df['ASD_Traits']

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#         # Step 4: Train the model
#         model = RandomForestClassifier(
#             n_estimators=5,
#             max_depth=3,
#             max_features=3,
#             min_samples_leaf=10,
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # Step 5: Make predictions
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         st.write(f"Model Accuracy: {accuracy:.2f}")
#         st.write("Classification Report:")
#         st.text(classification_report(y_test, y_pred))

#         # Step 6: Display feature importance
#         importances = model.feature_importances_
#         feature_names = X.columns
#         indices = np.argsort(importances)[::-1]

#         st.write("Feature Importances:")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.bar(range(X.shape[1]), importances[indices], align="center")
#         ax.set_xticks(range(X.shape[1]))
#         ax.set_xticklabels(feature_names[indices], rotation=90)
#         ax.set_xlabel("Features")
#         ax.set_ylabel("Importance")
#         st.pyplot(fig)

#         # Step 7: Allow user to input new data for prediction
#         st.sidebar.header("Input Features for Prediction")
#         input_data = {}
#         for col in X.columns:
#             input_data[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

#         # Predict using user input
#         input_df = pd.DataFrame([input_data])
#         prediction = model.predict(input_df)
#         st.write("Prediction for Input Data:")
#         st.write("ASD Traits" if prediction[0] == 1 else "No ASD Traits")

#     else:
#         st.error("The uploaded dataset does not have the target column 'Class/ASD Traits '.")
# else:
#     st.info("Please upload a dataset to proceed.")


