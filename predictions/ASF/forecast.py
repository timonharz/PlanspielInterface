import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data/all_stocks_5yr.csv")
data.head()

all_stock_tick_names = data['Name'].unique()
print(all_stock_tick_names)

stock_name = input("Stock Name: ")
all_data = data['Name'] == stock_name
final_data = data[all_data]
final_data.head()

final_data.plot('date', 'close', color="red")
new_data = final_data.head(60)
new_data.plot('date', 'close', color="green")
plt.show()

#Creating a new Dataframe and Training data

close_data = final_data.filter(['close'])
dataset = close_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
training_data_len = math.ceil(len(dataset) *0.7)
train_data = scaled_data[0:training_data_len, :]
x_train_data = []
y_train_data = []
for i in range(60,len(train_data)):
	x_train_data = list(x_train_data)
	y_train_data = list(y_train_data)
	x_train_data.append(train_data[i-60:i, 0])
	y_train_data.append(train_data[i, 0])
	x_train_data1, y_train_data1 = np.array(x_train_data), np.array(y_train_data)
	x_train_data2 = np.reshape(x_train_data1, (x_train_data1.shape[0], x_train_data1.shape[1], 1))

#Building LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences = True, input_shape=(x_train_data2.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compiling Model

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)

#Testing Model

test_data = scaled_data[training_data_len - 60: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
 
# 2.  Convert the values into arrays for easier computation
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
 
# 3. Making predictions on the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Error Calculation
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

#Making Predictions
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close')
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

