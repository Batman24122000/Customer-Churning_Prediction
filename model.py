# Import Libraries
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# load data
df = pd.read_csv('Final.csv')
X = df.drop('Exited', axis = 1)
y = df['Exited']

from sklearn.model_selection import train_test_split #to split data in 80-20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)#testing ke liye unknown data 20 % rakhna hai 
#random_state is used to get same data everytime when run
#we can give any value in random state
# For scaling data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model = joblib.load('model.pkl')


###### Load Model

def predict_churn(CustomerId,CreditScore,Geography,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary):
    x =np.zeros(len(X.columns))

    x[0]=CustomerId
    x[1]=CreditScore
    x[2]=Geography
    x[3]=Age
    x[4]=Tenure
    x[5]=Balance
    x[6]=NumOfProducts
    x[7]=HasCrCard
    x[8]=IsActiveMember
    x[9]=EstimatedSalary


    x = scaler.transform([x])[0]
    
    return model.predict([x])[0]