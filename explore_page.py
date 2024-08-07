import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Shorten Categories till the cutoff value
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# Convert Experience to Int
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 1
    return float(x)

# Clean Education
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

@st.cache_data
def load_data():
    #Get File Name and load file into dataframe
    filename = os.path.join(os.getcwd(), "data", "stackoverflow_survey.csv")
    df = pd.read_csv(filename)

    df = df[['Country','EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
    df = df[df['ConvertedCompYearly'].notnull()]
    df = df.dropna()
    df = df[df['Employment'].str.contains('Employed, full-time', case=False, regex=False)]
    df = df.drop('Employment', axis=1)
    
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)
    df = df[df['Country'] != 'Other']
    
    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel'].apply(clean_education)
    df = df.rename({'ConvertedCompYearly': 'Salary'}, axis=1)
    
    df = df[df['Salary'] <= 250000]
    df = df[df['Salary'] >= 10000]
    
    return df


df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
        ### Stack Overflow Developer Survey 2020
        """
    )

    data = df['Country'].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")

    st.write(
        """#### Number of Data from different countries"""
    )
    st.pyplot(fig1)

    st.write(
        """
        ### Mean Salary Based On Country
        """
    )
    data = df.groupby(['Country'])['Salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
        ### Mean Salary Based On Experience
        """
    )
    data = df.groupby(['YearsCodePro'])['Salary'].mean().sort_values(ascending=True)
    st.line_chart(data)

