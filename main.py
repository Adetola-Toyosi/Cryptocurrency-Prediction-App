# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pandas_datareader as pdd
import datetime as dt
import streamlit as st
import yfinance as yf
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Loading data .......................................................................................................
start = "2014-01-01"
end = dt.datetime.now()

st.title('Cryptocurrency Prediction App')

crypto_list = ('ETH-USD', 'USDT-USD', 'USDC-USD', 'BNB-USD', 'BUSD-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
               'DOGE-USD', 'DAI-USD', 'DOT-USD', 'WTRX-USD', 'HEX-USD', 'TRX-USD', 'MATIC-USD', 'AVAX-USD', 'SHIB-USD',
               'LTC-USD', 'BCH-USD', 'WBTC-USD', 'STETH-USD', 'UNI1-USD', 'LEO-USD', 'YOUC-USD',
               'FTT-USD')
select_crypto = st.selectbox('Select a crypto coin', crypto_list, key=str)


def load_data(crypto):
    data = yf.download(crypto, start, end)
    data.reset_index(inplace=True)
    return data


crypto = yf.Ticker(select_crypto)
story = crypto.info['description']
st.write(story)

date = crypto.history(period='max')
date = date.reset_index()
date = date['Date'][0]
st.write(f'{select_crypto} initially started on {date}')

data = load_data(select_crypto)

st.subheader(f'First 5 days prices of {select_crypto}')
st.write(data.head())
st.subheader(f'Prices from the last 5 days of {select_crypto}')
st.write(data.tail())

# Plotting the moving Average ..........................................................................................
st.subheader(f'Moving Average of {select_crypto} coin')
MA5 = data.Close.rolling(50).mean()
MA100 = data.Close.rolling(100).mean()

fig1 = plt.figure(figsize=(20, 10))

plt.plot(data.Close, color='blue', label='Closing prices')
plt.plot(MA5, 'red', label='Moving Average for 50 days')
plt.plot(MA100, 'black', label='Moving Average for 100 days')

plt.xlabel('Time', fontsize=17)
plt.ylabel(f'{select_crypto} closing price', fontsize=17)
plt.legend(fontsize=20)
plt.tight_layout()

st.pyplot(fig1)

# scaling the data .....................................................................................................
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
scale_load_state = st.text('Scaling data...')
data = load_data(select_crypto)
scale_load_state.text(f'{select_crypto} dataset has been scaled')


# Deep learning ........................................................................................................
def lstm_model(select_crypto):
    model = Sequential()

    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    return model


model_load_state = st.text(f'Training {select_crypto} data...\nThis would be done shortly')
model = lstm_model(load_data(select_crypto))
model_load_state.text(f'{select_crypto} dataset has been trained')

# preparing test data ..................................................................................................
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = pdd.DataReader(f'{select_crypto}', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# making predictions ...................................................................................................
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# for plotting the predictions .........................................................................................
st.subheader(f'{select_crypto} Predictive Analysis')
fig = plt.figure(figsize=(40, 20))
plt.plot(actual_prices, color='brown', label='Real closing prices')
plt.plot(prediction_prices, color='black', label='Predicted closing prices')
plt.xlabel('Time', fontsize=30)
plt.ylabel('Price', fontsize=30)
plt.legend(fontsize=30)
plt.grid(alpha=0.4)
st.pyplot(fig)

# for predicting the next day ..........................................................................................
st.subheader('Prediction')


def next_day():
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


st.write(f'The predicted price of {select_crypto} tomorrow would be: ${next_day()[0][0]:.3f}')
