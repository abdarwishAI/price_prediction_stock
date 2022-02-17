# Import required libraries
import numpy as np
#import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
import yfinance as yf
import math
import datetime
import joblib
from tensorflow import keras
#import os
from datetime import date
from datetime import datetime, timedelta
import streamlit as st


# Get the Stock price
import datetime
from datetime import date
today = date.today()
end_date = today
start_date = '2010-12-12'

# Get the data
#data = yf.download('AAPL', start_date, end_date)


#### Streamlit Web App
st.title('Stock Price Prediction')

Ticker = st.text_input("Enter Stock Symbol 'Ticker' examples:   FB  ,  MSFT  ,  TWTR ,  AAPL " , "TWTR")
df = yf.download(Ticker, start_date, end_date)
st.subheader(f'{Ticker} Stock Dataset')
df_table = df.reset_index()
df_table['Day'] = pd.to_datetime(df_table['Date']).dt.date
df_table1 = df_table.sort_values('Day', ascending=False)
df_table2 = df_table1[['Day','Open','High','Low','Close']]
#df_table['Day'] = pd.to_datetime(df_table['Date']).dt.date
st.write(df_table2)

st.subheader(f"Closing Price Trend for {Ticker} vs. Date")
#visualize stock plus the Moving Average
fig = plt.figure(figsize=(12,6))
plt.title('Close Price Trend for the Stock', color = 'black')
plt.plot(df['Close'], color = 'blue')
plt.xlabel('Date', fontsize=18, color = 'black')
plt.ylabel('Price (USD)', fontsize=18, color = 'black')
st.pyplot(fig)

# Draw MA 20,50 indicator
ma20 = df.Close.rolling(50).mean()
ma50 = df.Close.rolling(100).mean()
st.subheader(f"Closing Price Trend for {Ticker} vs. Date with Moving Average indicators")
fig2 = plt.figure(figsize=(12,6))
plt.title(f'Close Price Trend for {Ticker} with Moving Average indicators', color = 'black')
plt.plot(df.Close)
plt.plot(ma20, 'g') # red color
plt.plot(ma50, 'r') # color = green
plt.xlabel('Date', fontsize=18, color = 'black')
plt.ylabel('Price (USD)', fontsize=18, color = 'black')
plt.legend(['Close', 'MA50', 'MA100'], loc ='upper left')
st.pyplot(fig2)


# create df with only close price feature
data = df.filter(['Close'])
# convert the data to np.array
dataset = data.values
# Get Training data 70 %
training_data_len = math.ceil(len(dataset)* 0.70)

# Scaling the data using saved Scaler
scaler = joblib.load('my_scaler.gz')

# Transform the data
scaled_data = scaler.fit_transform(dataset)

# Load ML Model
#model_file = os.getcwd() + '/' + 'Price_prediction.h5'
my_model = keras.models.load_model('Price_V3.h5')

# Create test dataset
test_data = scaled_data[training_data_len - 60:, :]
# Split x_test, y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# convert x_test into np array
x_test = np.array(x_test)
# re-shape x_test from 2D to 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict x_test
predictions = my_model.predict(x_test)
# convert back the predicted values into original data "reverse MinMax scaler"
predictions = scaler.inverse_transform(predictions)

# Evaluate the model using RMSE
def Model_evaluation(prediction, y_test):
    rmse = np.sqrt(np.mean(prediction - y_test) ** 2)
    return rmse
Model_evaluation(predictions, y_test)


# plot the data
# rename test by actual
train = data[:training_data_len]
actual = data[training_data_len:]
actual['predictions'] = predictions
# Plot
st.subheader(f" Predicting Closing Price for {Ticker} Actual vs Predicted Price")
fig3 = plt.figure(figsize=(12,6))
plt.title('Actual Price vs. Predicted Price', color = 'black')
plt.plot(train.Close)
plt.plot(actual.Close, 'green') # red color
plt.plot(actual.predictions, 'red')
#plt.plot(ma50, 'r') # color = green
plt.xlabel('Date', fontsize=18, color = 'black')
plt.ylabel('Price (USD)', fontsize=18, color = 'black')
plt.legend(['Train', 'Actual_Price', 'predicted_Price'], loc ='upper left')
st.pyplot(fig3)

st.subheader('Actual Closing Price vs Predicted Price')
actual1 = actual.reset_index()
actual2 = actual.sort_values('Date', ascending=False)
actual2.head()
st.write(actual2.head())



############### Now Lets Compare Last days data
today = date.today()
yesterday = date.today() - timedelta(1)
lastdays = date.today() - timedelta(7)
start_date = '2010-12-12'
end_date = yesterday
ticker = Ticker
dfy1 = yf.download(ticker, start_date, end_date)
dfy2 = dfy1.filter(['Close'])

previous_days = dfy2[-60:].values
#my_scaler = joblib.load('my_scaler.gz')
previous_days_scaled = scaler.transform(previous_days)
x_test2 = []
x_test2.append(previous_days_scaled)
x_test2 = np.array(x_test2)
x_test2 = np.reshape(x_test2, (x_test2.shape[0],x_test2.shape[1], 1 ))
#model_file = os.getcwd() + '/' + 'Price_prediction.h5'
my_model = keras.models.load_model('Price_V3.h5')
predict_Last_price = my_model.predict(x_test2)
predict_Last_price = scaler.inverse_transform(predict_Last_price)

#check the actual price for yesterday
start_date = lastdays
end_date = today
dfy1 = yf.download(ticker, start_date, end_date)
dfy2 = dfy1.filter(['Close'])

#get the values
dfy4 = dfy2.sort_values('Date', ascending=False)
last_closing_price = dfy4.Close[0]
dfy3 = dfy4.reset_index()
dfy3['Day'] = pd.to_datetime(dfy3['Date']).dt.date
last_date = str(dfy3.Day[0])

st.subheader(f" Test on Last Day Closing Price")
st.write(f'Last Day Date = {last_date}')
#print (f'Last Day Date = {last_date}')
st.write (f'Last Day Actual Closing Price = [[{last_closing_price}]]')
st.write (f'Last Day Predicted Closing Price = {predict_Last_price}')


###### predict today price
import datetime
from datetime import date
today = date.today()
start_date = '2010-12-12'
end_date = today
ticker = Ticker
df1 = yf.download(ticker, start_date, end_date)
df2= df1.filter(['Close'])

previous_days = df2[-60:].values
#my_scaler = joblib.load('my_scaler.gz')
previous_days_scaled = scaler.transform(previous_days)
x_test2 = []
x_test2.append(previous_days_scaled)
x_test2 = np.array(x_test2)
x_test2 = np.reshape(x_test2, (x_test2.shape[0],x_test2.shape[1], 1 ))
#model_file = os.getcwd() + '/' + 'Price_prediction.h5'
my_model = keras.models.load_model('Price_V3.h5')
predict_future_price = my_model.predict(x_test2)
predict_future_price = scaler.inverse_transform(predict_future_price)


st.subheader(f" Predecting New Day Closing Price")
st.write (f'Today Date = {today}')
st.write (f'Predicted closing price = {predict_future_price}')
st.write ("")
st.write ("___________________________________________________________")
st.write ("")
st.write ("") 
