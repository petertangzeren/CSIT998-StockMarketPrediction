# 注意：老菜用这个版本，这个是最终版
import numpy as np
from keras import backend
import pickle
import pandas as pd
import os, datetime
from tensorflow import keras
import tensorflow as tf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from transformer_class import create_model

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


def train(model_path, df):
    # Set bars for later splitting
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1 * len(times))]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2 * len(times))]  # Last 20% of series

    # Normalise (Can be changed to use pct_change way to normalise by Cai)
    scaler_1 = MinMaxScaler()
    df = scaler_1.fit_transform(df)
    df = pd.DataFrame(df, columns=['Close'])

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
    training = model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_val, y_val))

    # 下面的保存模型，还有show plot照搬了老菜的那部分
    model.save(model_path)
    scaler_path = os.path.join(model_path, "scaler.pkl")
    loss_acc = model.evaluate(X_test, y_test, verbose=0)
    print("test loss, test acc:", loss_acc)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_1, f)
    plt.plot(training.history["loss"])
    plt.plot(training.history["val_loss"])
    plt.show()

    return model, scaler_1


def predict(model_path, prices, predicting_days=1):
    scaler_path = os.path.join(model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaled_prices = scaler.transform(prices)
    x_input = scaled_prices[-100:].T

    model = keras.models.load_model(model_path)
    y_predict = model.predict(x_input)
    future = scaler.inverse_transform(y_predict).ravel()[0]
    daily_prediction = (1 + future) ** (1 / predicting_days) - 1

    prediction_series = [daily_prediction] * predicting_days
    return np.array(prediction_series)


# def predict_price(model, scaler, prices, predicting_days=1):
#     scaled_prices = scaler.transform(prices)
#     x_input = scaled_prices[-100:].T
#
#     y_predict = model.predict(x_input)
#     future = scaler.inverse_transform(y_predict).ravel()[0]
#     daily_prediction = (1 + future) ** (1 / predicting_days) - 1
#
#     prediction_series = [daily_prediction] * predicting_days
#     return np.array(prediction_series)


def main():
    ticker = "MSFT"
    df = pdr.get_data_tiingo(ticker, api_key="aeeaa9dbc8f82f2c361abaa259050d75e736b424")
    df.to_csv("MSFT.csv")
    df = pd.read_csv("MSFT.csv", delimiter=',', usecols=['date', 'close'])
    df.sort_values('date', inplace=True)
    df = df[["close"]].pct_change().dropna()

    model_path = f"transformer_models/{ticker}_close_transformer"

    train(model_path,df)
    prediction = predict(model_path,  df, predicting_days=predict_days)
    print(prediction)


if __name__ == "__main__":
    main()