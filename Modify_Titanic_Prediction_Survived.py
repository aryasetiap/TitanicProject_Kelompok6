import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data
@st.cache
def load_data():
    return pd.read_csv('train.csv')

# Function to preprocess data
def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
    
    return df

# Function to train model
def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    return model

# Function to display statistical summary
def display_summary(df):
    st.write('Number of rows and columns:', df.shape)
    st.write('Statistical measures of the data:')
    st.write(df.describe())

# Function to display count plots
def display_count_plots(df):
    st.title('Titanic Data Analysis')

    fig_survived = plt.figure(figsize=(8, 5))
    sns.countplot(x='Survived', data=df)
    plt.title('Count Plot of Survived')
    st.pyplot(fig_survived)

    fig_sex = plt.figure(figsize=(8, 5))
    sns.countplot(x='Sex', data=df)
    plt.title('Count Plot of Sex')
    st.pyplot(fig_sex)

    fig_sex_survived = plt.figure(figsize=(8, 5))
    sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title('Survival Count by Gender')
    st.pyplot(fig_sex_survived)

    fig_pclass = plt.figure(figsize=(8, 5))
    sns.countplot(x='Pclass', data=df)
    plt.title('Count Plot of Pclass')
    st.pyplot(fig_pclass)

    fig_pclass_survived = plt.figure(figsize=(8, 5))
    sns.countplot(x='Pclass', hue='Survived', data=df)
    plt.title('Survival Count by Pclass')
    st.pyplot(fig_pclass_survived)

# Function to make predictions
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    return prediction, prediction_proba

# Main function
def main():
    st.title('Titanic Survival Prediction')
    
    # Load data
    titanic_data = load_data()
    
    # Preprocess data
    titanic_data = preprocess_data(titanic_data)
    
    # Display summary statistics
    display_summary(titanic_data)
    
    # Split data into features and target variable
    X = titanic_data.drop(columns=['Survived'], axis=1)
    Y = titanic_data['Survived']
    
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    # Train the model
    model = train_model(X_train, Y_train)
    
    # Display count plots
    display_count_plots(titanic_data)
    
    # Model evaluation
    Y_train_pred = model.predict(X_train)
    training_accuracy = accuracy_score(Y_train, Y_train_pred)
    Y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    
    # Display accuracy scores
    st.subheader('Model Evaluation - Accuracy Scores')
    st.write('Accuracy score on training data:', training_accuracy)
    st.write('Accuracy score on test data:', test_accuracy)
    
    # Predict survival for a new passenger
    st.subheader('Predict Titanic Survival for a New Passenger')
    
    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 80, 30)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.number_input('Number of Parents/Children Aboard', 0, 6, 0)
    fare = st.slider('Fare', 0, 500, 50)
    embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])
    
    sex = 0 if sex == 'male' else 1
    embarked = 0 if embarked == 'S' else 1 if embarked == 'C' else 2
    
    if st.button('Predict Survival'):
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction, prediction_proba = make_prediction(model, input_data)
        
        if prediction[0] == 1:
            st.success(f'The passenger is likely to survive with a probability of {prediction_proba[0][1]:.2f}')
        else:
            st.error(f'The passenger is unlikely to survive with a probability of {prediction_proba[0][0]:.2f}')

if __name__ == '__main__':
    main()
