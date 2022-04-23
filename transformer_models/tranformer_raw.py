import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from transformer_models.transformer_class import create_model

print('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

batch_size = 32
seq_len = 100

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

# Predict for 5 days later
predict_days = 5


def train(df):
    # Set bars for later splitting
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1 * len(times))]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2 * len(times))]  # Last 20% of series

    # Normalise (Can be changed to use pct_change way to normalise by Cai)
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    df = scaler_1.fit_transform(np.array(df).reshape(-1, 1))
    df = pd.DataFrame(df, columns=['Close'])

    # Drop all 0 rows
    df.replace(0, np.nan, inplace=True)
    df.dropna(how='any', axis=0, inplace=True)
    # df.reset_index(inplace=True)

    # Split dataset to training (80%), validation(10%) and testing(10%)
    df_train = df[(df.index < last_20pct)]
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values

    # Split all datasets into chunks for further model training (Chunk length 100)
    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)-predict_days):
        X_train.append(train_data[i - seq_len:i])
        y_train.append(train_data[:, 0][i+predict_days])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)-predict_days):
        X_val.append(val_data[i - seq_len:i])
        y_val.append(val_data[:, 0][i+predict_days])
    X_val, y_val = np.array(X_val), np.array(y_val)

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)-predict_days):
        X_test.append(test_data[i - seq_len:i])
        y_test.append(test_data[:, 0][i+predict_days])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Create a transformer model
    model = create_model()
    model.summary()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=35, validation_data=(X_val, y_val))

    return model, scaler_1


def predict(model, df_close, scaler, days):
    df_close = np.array(df_close).reshape(-1, 1)

    train_size = int(len(df_close) * 0.8)
    close_test = df_close[train_size : len(df_close)]

    num_take = len(close_test) - 100
    x_input = close_test[num_take:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while i < 60:

        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    all_pred = scaler.inverse_transform(lst_output[:days])

    return all_pred


def main():
    df = pdr.get_data_tiingo("MSFT", api_key="aeeaa9dbc8f82f2c361abaa259050d75e736b424")
    df.to_csv("MSFT.csv")
    df = pd.read_csv("MSFT.csv", delimiter=',', usecols=['date', 'close'])
    df.sort_values('date', inplace=True)
    df = df["close"]

    model, scaler = train(df)
    pred_values = predict(model, df, scaler, 30)
    print(pred_values)



if __name__ == "__main__":
    main()