import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dataset
train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=train.iloc[:,1:2].values
print(len(train))
# Feautre Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_scaled=sc.fit_transform(training_set)
training_set.shape
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_scaled[i-60:i,0])
    y_train.append(training_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
print(X_train.shape)

#Reshaping
X_train=np.reshape(X_train,(1198, 60,1))

#Part-2 Building the RNN
#Importing the Keras Libraries and Packages.
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)
#reading testing dataset
test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test.iloc[:,1:2].values
#step-3
#predicting stock prices from model
dataset_total=pd.concat((train['Open'],test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)



X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(20,60,1))

prediction=model.predict(X_test)


plt.plot(real_stock_price,color='red',label='real_stock_price')
plt.plot(prediction,color='red',label='real_stock_price')


plt.xlabel('Time')
plt.ylabel('Stock _price')
plt.title('Stock_price_prediction')

plt.show()




















