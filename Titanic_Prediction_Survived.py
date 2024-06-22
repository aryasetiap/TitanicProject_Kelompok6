import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from csv file to Pandas DataFrame
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')


titanic_data = load_data()

# Handling Missing Values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(
    titanic_data['Embarked'].mode()[0], inplace=True)

# Encoding Categorical Columns
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
                     'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Separating features & Target
X = titanic_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

# Splitting the data into training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

# Model Training - Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Model Evaluation - Accuracy Score
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)

Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

# Streamlit App
st.title('Titanic Survival Prediction')
st.write('Number of rows and columns:', titanic_data.shape)
st.write('Statistical measures of the data:')
st.write(titanic_data.describe())

# Data Visualization
st.title('Titanic Data Analysis')

# Count plot for "Survived" column
fig_survived = plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=titanic_data)
plt.title('Count Plot of Survived')
st.pyplot(fig_survived)

# Count plot for "Sex" column
fig_sex = plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', data=titanic_data)
plt.title('Count Plot of Sex')
st.pyplot(fig_sex)

# Count plot for "Sex" column with hue "Survived"
fig_sex_survived = plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Gender')
st.pyplot(fig_sex_survived)

# Count plot for "Pclass" column
fig_pclass = plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', data=titanic_data)
plt.title('Count Plot of Pclass')
st.pyplot(fig_pclass)

# Count plot for "Pclass" column with hue "Survived"
fig_pclass_survived = plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival Count by Pclass')
st.pyplot(fig_pclass_survived)

# Display Accuracy Scores
st.subheader('Model Evaluation - Accuracy Scores')
st.write('Accuracy score on training data:', training_accuracy)
st.write('Accuracy score on test data:', test_accuracy)
