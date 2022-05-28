import os
import pickle
import pandas as pd
import numpy as np
from keras import backend
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import csv
from transformer_class import Time2Vector
from transformer_class import SingleAttention
from transformer_class import MultiAttention
from transformer_class import TransformerEncoder
from transformer_class import create_model

import matplotlib.pyplot as plt

batch_size = 32
seq_len = 128  # Size of x chunks
shift_num = 4  # Prediction Days

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256


def seperate_data(dataset, seq_len=128, shift_num=4):
    dataX, dataY = [], []
    for i in range(seq_len, len(dataset) - shift_num):
        dataX.append(dataset[i - seq_len : i])
        dataY.append(dataset[:, 0][i + shift_num])
    dataX, dataY = np.array(dataX), np.array(dataY)
    return dataX, dataY


def seperate_x(dataX, shift_num=4, seq_len=128):
    x_chunks = []
    for i in range(0, shift_num + 1):
        x_chunks.append(dataX[i : i + seq_len])

    x_chunks = np.array(x_chunks)
    return x_chunks


def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def train(model_path, prices, seq_len=128):
    times = sorted(prices.index.values)
    last_10pct = sorted(prices.index.values)[
        -int(0.1 * len(times))
    ]  # Last 10% of series
    last_20pct = sorted(prices.index.values)[
        -int(0.2 * len(times))
    ]  # Last 20% of series
    last_30pct = sorted(prices.index.values)[
        -int(0.3 * len(times))
    ]  # Last 30% of series

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    scaled_prices = pd.DataFrame(scaled_prices, columns=["Close"])
    scaled_prices.replace(0, np.nan, inplace=True)
    scaled_prices.dropna(how="any", axis=0, inplace=True)

    df_train = scaled_prices[
        (scaled_prices.index < last_30pct)
    ]  # Training data are 70% of total data
    df_val = scaled_prices[
        (scaled_prices.index >= last_30pct) & (scaled_prices.index < last_20pct)
    ]
    df_test = scaled_prices[(scaled_prices.index >= last_20pct)]

    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values

    x_train, y_train = seperate_data(train_data)
    x_val, y_val = seperate_data(val_data)
    x_test, y_test = seperate_data(test_data)

    # Create a transformer model
    model = create_model()
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint(
        "Transformer+TimeEmbedding.hdf5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=15,
        callbacks=[callback],
        validation_data=(x_val, y_val),
    )

    model = tf.keras.models.load_model(
        "Transformer+TimeEmbedding.hdf5",
        custom_objects={
            "Time2Vector": Time2Vector,
            "SingleAttention": SingleAttention,
            "MultiAttention": MultiAttention,
            "TransformerEncoder": TransformerEncoder,
        },
    )

    model.save(model_path)
    scaler_path = os.path.join(model_path, "scaler.pkl")
    loss_acc = model.evaluate(x_test, y_test, verbose=0)
    print("test loss, test acc:", loss_acc)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # plt.plot(history.history["loss"])
    # plt.plot(history.history["val_loss"])
    # plt.show()


def predict(model_path, prices, seq_len=128):
    scaler_path = os.path.join(model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaled_prices = scaler.transform(np.array(prices).reshape(-1, 1)).reshape(
        len(prices)
    )
    x_input = scaled_prices[-seq_len - shift_num :]
    x_input = seperate_x(x_input)

    model = keras.models.load_model(model_path)
    y_predict = model.predict(x_input)
    future = scaler.inverse_transform(y_predict)
    return future


def main():
    import pandas_datareader as pdr

    stock_name = "MSFT"
    ticker = yf.Ticker(stock_name)
    df = ticker.history(period="max")
    df.to_csv("MSFT.csv")
    df = pd.read_csv("MSFT.csv", delimiter=",", usecols=["Date", "Close"])

    df.sort_values("Date", inplace=True)
    df[["Close"]] = df[["Close"]].rolling(10).mean()
    df.dropna(how="any", axis=0, inplace=True)

    prices = df
    prices["Close"] = prices["Close"].pct_change()  # Create arithmetic returns column
    prices.dropna(how="any", axis=0, inplace=True)
    prices.drop(columns=["Date"], inplace=True)
    prices = prices["Close"]

    model_path = f"transformer_models/{stock_name}_close"

    train(model_path, prices)
    prediction = predict(model_path, prices)
    print(prediction)


if __name__ == "__main__":
    main()
