from flask import Flask, render_template, request
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    df = pd.read_csv("student_spending.csv")
    df = df.drop("Unnamed: 0", axis=1)

    X = df.drop(columns=['preferred_payment_method','gender','year_in_school','major'])
    y_value = df['preferred_payment_method']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_value)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get data from the form
    data = [float(request.form['age']),
            float(request.form['health_wellness']),
            float(request.form['monthly_income']),
            float(request.form['food'])]

    # Make prediction
    prediction = model.predict([data])

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
