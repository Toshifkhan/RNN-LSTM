# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:40:18 2020

@author: toshif
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1], 1))

#port 2: Building the rnn

#importing the keras libraries and package\

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout

#Intiallising the RNN
regressor=Sequential()

#Adding the first LSTM layer and some Droptout regularisation
regressor.add(LSTM(units =50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the secound lstm layer and some droptout layer

regressor.add(LSTM(units =50,return_sequences=True))
regressor.add(Dropout(0.2))


#Adding a third LSTM layer and some droptout regularisation.

regressor.add(LSTM(units =50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some droptout regularisation 

regressor.add(LSTM(units =50))
regressor.add(Dropout(0.2))


#Part 3 :-
#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#fitting the Rnn to the training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)



#part 3:    #making the prediction and visualising the results'

#getting the real price of stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price =dataset_test.iloc[:,1:2].values


#geting the pridicted stock price of 2017
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs =inputs.reshape(-1,1)
inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualising the results:
plt.plot(real_stock_price,color='red',label='Real Google stock price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google')
plt.title('Google stock price pridiction')
plt.xlabel('Time')
plt.ylabel('Google stock Price')
plt.legend()
























