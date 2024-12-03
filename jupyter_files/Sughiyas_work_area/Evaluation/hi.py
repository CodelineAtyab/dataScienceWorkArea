import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    df = pd.read_csv("insurance.csv")
    # st.dataframe(df)
    data = df

    st.title("Insurance Data Visualization")
    st.subheader("Scatter Plot: Age vs. Charges")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', data=data, hue='smoker')
    plt.title("Charges by Age and Smoking Status")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    st.pyplot(plt.gcf())

       # Box plot: Charges by Smoker and BMI
    st.subheader("Box Plot: Charges by Smoker and BMI")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='smoker', y='charges', data=data, hue='sex')
    plt.title("Charges by Smoking Status and Gender")
    plt.xlabel("Smoker")
    plt.ylabel("Charges")
    st.pyplot(plt.gcf())

    
if __name__=="__main__":
    main()
