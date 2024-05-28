import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Function to load and process data
def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                st.error("The file encoding is not supported or the file is empty. Please upload a valid CSV file with UTF-8 or ISO-8859-1 encoding.")
                return None
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a non-empty CSV file.")
            return None
        return df
    return None

def preprocess_data(df):
    st.write("## Dataset Preview")
    st.write(df.head())
    st.write("## Statistical Overview")
    st.write(df.describe())
    st.write("## Missing Values")
    st.write(df.isnull().sum())

    st.write("## Handle Missing Values")
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            st.write(f"Column: {column}")
            action = st.selectbox(f"Choose action for {column}", 
                                  options=["None", "Remove Rows", "Replace with Mean", "Replace with Median", "Replace with Mode"], 
                                  key=column)
            if action == "Remove Rows":
                df = df.dropna(subset=[column])
            elif action == "Replace with Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif action == "Replace with Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif action == "Replace with Mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def convert_data_types(df):
    st.write("## Data Type Conversion")
    for column in df.columns:
        st.write(f"Column: {column}, Current Data Type: {df[column].dtype}")
        new_type = st.selectbox(f"Convert {column} to", options=["None", "int", "float", "str"], key=f"convert_{column}")
        if new_type != "None":
            try:
                if new_type == 'str':
                    df[column] = df[column].astype(str)
                else:
                    df[column] = df[column].astype(new_type)
            except ValueError:
                st.warning(f"Conversion of {column} to {new_type} failed. Please choose a compatible data type.")
    st.write("## Final Dataset Columns with Data Types")
    st.write(df.dtypes)
    return df

def show_standard_deviation(df):
    st.write("## Standard Deviation of Columns")
    method = st.selectbox("Calculate Standard Deviation with respect to", ["Mean", "Median", "Mode"])
    std_devs = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == "Mean":
            std_dev = np.std(df[column])
        elif method == "Median":
            std_dev = np.std(df[column] - df[column].median())
        elif method == "Mode":
            std_dev = np.std(df[column] - df[column].mode()[0])
        std_devs[column] = std_dev
    sorted_std_devs = dict(sorted(std_devs.items(), key=lambda item: item[1]))
    st.write(sorted_std_devs)

def encode_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def train_models(X, y, task):
    models = {}
    if task == 'Classification':
        models['Logistic Regression'] = LogisticRegression()
        models['Decision Tree'] = DecisionTreeClassifier()
        models['Random Forest'] = RandomForestClassifier()
    else:
        models['Linear Regression'] = LinearRegression()
        models['Decision Tree'] = DecisionTreeRegressor()
        models['Random Forest'] = RandomForestRegressor()
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    return trained_models

def evaluate_models(trained_models, X_test, y_test, task):
    evaluations = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if task == 'Classification':
            evaluations[name] = accuracy_score(y_test, y_pred)
        else:
            evaluations[name] = mean_squared_error(y_test, y_pred, squared=False)
    return evaluations

def visualize_data(df):
    st.write("Data Preview:")
    st.dataframe(df.head())
    columns = df.columns.tolist()
    if columns:
        chart_type = st.selectbox("Select chart type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Pie Chart"])
        if chart_type in ["Histogram", "Bar Plot", "Pie Chart"]:
            x_axis = st.selectbox("Select column", columns)
        else:
            x_axis = st.selectbox("Select X-axis column", columns)
            y_axis = st.selectbox("Select Y-axis column", columns)
        plt.figure(figsize=(10, 6))
        if chart_type == "Scatter Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Line Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.lineplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Bar Plot":
            st.write(f"{chart_type} of {x_axis}")
            sns.countplot(data=df, x=x_axis)
        elif chart_type== "Histogram":
            st.write(f"{chart_type} of {x_axis}")
            sns.histplot(df[x_axis], kde=True)
        elif chart_type == "Box Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.boxplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Pie Chart":
            st.write(f"Pie Chart of {x_axis}")
            df[x_axis].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.ylabel('')
        st.pyplot(plt)
    else:
        st.error("The CSV file does not contain any columns.")

def main():
    fixed_string = "Developer -> Satwik Tiwari\REC BIJNOR(IT)\n from Prayagraj."

    # Display the fixed string at the end of the page
    st.write(fixed_string)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Preprocess", "Preprocessing", "Visualization", "Model Training", "Predict"])

    if page == "Upload & Preprocess":
        st.title("Upload & Preprocess CSV Data")
        data = load_data()
        if data is not None:
            data = preprocess_data(data)

    elif page == "Preprocessing":
        st.title("Data Preprocessing")
        data = load_data()
        if data is not None:
            data = preprocess_data(data)
            data = convert_data_types(data)
            st.write("## Column Correlation with Target")
            target_column = st.selectbox("Select Target Column", options=data.columns)
            if target_column:
                numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
                if target_column in numeric_columns:
                    corr_method = st.selectbox("Select Correlation Method", options=["Pearson", "Mutual Information"])
                    if corr_method == "Pearson":
                        correlations = data[numeric_columns].corr()[target_column].sort_values(ascending=False)
                    elif corr_method == "Mutual Information":
                        if data[target_column].dtype == 'object':
                            correlations = pd.Series(mutual_info_classif(data[numeric_columns].drop(columns=[target_column]), data[target_column]), index=data[numeric_columns].drop(columns=[target_column]).columns)
                        else:
                            correlations = pd.Series(mutual_info_regression(data[numeric_columns].drop(columns=[target_column]), data[target_column]), index=data[numeric_columns].drop(columns=[target_column]).columns)
                        correlations = correlations.sort_values(ascending=False)
                    st.write(correlations)
                else:
                    st.error("Selected target column is not numeric. Correlation analysis can only be performed on numeric columns.")
                drop_columns = st.multiselect("Select columns to drop", options=data.columns)
                if st.button("Drop Selected Columns"):
                    data.drop(columns=drop_columns, inplace=True)
                    st.write("Selected columns dropped successfully.")

    elif page == "Visualization":
        st.title("CSV Data Visualization")
        data = load_data()
        if data is not None:
            visualize_data(data)

    elif page == "Model Training":
        st.title("Train Machine Learning Models")
        data = load_data()
        if data is not None:
            data = preprocess_data(data)
            input_columns = st.multiselect("Select Input Columns", options=data.columns)
            target_column = st.selectbox("Select Target Column", options=data.columns)
            if input_columns and target_column:
                X = data[input_columns]
                y = data[target_column]
                X, _ = encode_data(X)
                y, label_encoder = encode_data(pd.DataFrame(y))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                task = st.radio("Select Task Type", ('Classification', 'Regression'))
                trained_models = train_models(X_train, y_train.values.ravel(), task)
                evaluations = evaluate_models(trained_models, X_test, y_test.values.ravel(), task)
                st.write("## Model Performance")
                st.write(evaluations)
                selected_model = st.selectbox("Select Model", options=list(trained_models.keys()))
                st.session_state["selected_model"] = trained_models[selected_model]
                st.session_state["input_columns"] = input_columns
                st.session_state["label_encoder"] = label_encoder[target_column] if task == 'Classification' else None
    elif page == "Predict":
        st.title("Make Predictions")
        if "selected_model" in st.session_state:
            model = st.session_state["selected_model"]
            input_columns = st.session_state["input_columns"]
            label_encoder = st.session_state.get("label_encoder", None)
            user_input = {}
            for col in input_columns:
                user_input[col] = st.text_input(f"Input {col}")
            if st.button("Predict"):
                user_input_df = pd.DataFrame([user_input])
                user_input_df, _ = encode_data(user_input_df)
                prediction = model.predict(user_input_df)
                if label_encoder:
                    prediction = label_encoder.inverse_transform(prediction)
                st.write(f"Prediction: {prediction}")
        else:
            st.error("Please train a model first in the 'Model Training' page.")

if __name__ == '__main__':
    main()

