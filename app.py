# Import Libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import model

app = Flask(__name__)


# render html page
@app.route('/')
def home():
    return render_template('index.html')


# get user input and the predict the output and return to user
@app.route('/predict', methods=['POST'])
def predict():
    # take data from form and store in each feature
    input_features = [x for x in request.form.values()]
    CustomerId = input_features[0]
    CreditScore = input_features[1]
    Geography = input_features[2]
    Age = input_features[3]
    Tenure = input_features[4]
    Balance = input_features[5]
    NumOfProducts = input_features[6]
    HasCrCard = input_features[7]
    IsActiveMember = input_features[8]
    EstimatedSalary = input_features[9]


# predict the price of house by calling model.py
    predicted_churner = model.predict_churn(CustomerId,CreditScore,Geography,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary)

    if(int(predicted_churner)==1):
        prediction='Churn'
    else:
        prediction='Not Churn' 

# render the html page and show the output
    return render_template('index.html', prediction_text='The customer will {}'.format(prediction))

# if __name__ == "__main__":
#   app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    app.run(debug=True)