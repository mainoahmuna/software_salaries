import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('salaryPredModel.pkl', 'rb') as file:
        data = pickle.load(file)

    return data

data = load_model()
model = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        'United States of America',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Canada',
        'India',
        'France',
        'Brazil',
        'Netherlands',
        'Australia',
        'Spain',
        'Poland',
        'Sweden',
        'Italy',
        'Switzerland',
        'Denmark',
        'Norway',
        'Israel',
        'Portugal',
        'Austria'
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country",countries)
    educatio_lvl = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        user_input = np.array([[country, educatio_lvl, experience]])
        user_input[:, 0] = le_country.transform(user_input[:, 0])
        user_input[:, 1] = le_education.transform(user_input[:, 1])
        user_input = user_input.astype(float)

        salary = model.predict(user_input)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
