import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)
def app():
    st.title("🩺 Diabetes Prediction ")
    img_url = "https://d3upjtc0wh66ez.cloudfront.net/wp-content/uploads/2024/09/diabetes-symptoms-and-treatment.jpg"
    col1, col2, col3 = st.columns([1, 3, 1])
    st.sidebar.header("Input Patient Data")
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 300, 120)
    bloodpressure = st.sidebar.slider("Blood Pressure", 0, 200, 70)
    skinthickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 20.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 1, 120, 25)
    input_data = np.array([pregnancies, glucose, bloodpressure, skinthickness,
                           insulin, bmi, dpf, age]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    if st.button("Predict Diabetes"):
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("This person has diabetes")
        else:
            st.success("This person does not have diabetes")
if __name__ == "__main__":
    app()
