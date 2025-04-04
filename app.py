import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(
    open("trained_model.sav", 'rb'))


def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "Patient is non-diabetic"
    else:
        return "Patient is diabetic"


def main():
    # giving a title
    st.title('Diabetes Prediction Web App')

    # getting the input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction value')
    Age = st.text_input('Age of the person')

    # code for prediction
    diagnosis = ''

    # create a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()

