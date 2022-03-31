import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import losses


def seperate_data(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# 三个parametres
# model_path：模型路径
# df_close：所有的close数据
# scaler：把code中最新的scaler继承过来就行
# days：需要预测的天数
def stock_predict(model_path, df_close, scaler, days):
    model = keras.models.load_model(model_path)

    train_size = int(len(df_close) * 0.7)
    close_test = df_close[train_size : len(df_close), :1]

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
    # df = pdr.get_data_tiingo("600958", api_key="aeeaa9dbc8f82f2c361abaa259050d75e736b424")
    df.to_csv("MSFT.csv")
    df_stock = pd.read_csv("MSFT.csv")

    df_close = df_stock["close"]
    df_date = (df_stock["date"].str[:4] + df_stock["date"].str[5:7]).astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

    train_size = int(len(df_close) * 0.7)
    test_size = len(df_close) - train_size
    close_train = df_close[0:train_size]
    close_test = df_close[train_size : len(df_close), :1]

    x_train, y_train = seperate_data(close_train, 100)
    x_test, y_test = seperate_data(close_test, 100)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # create the model
    model = models.Sequential()

    model.add(layers.LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(1))

    img_new = model.predict(x_test)

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=64,
        verbose=1,
    )

    change_r = stock_predict(
        "/Users/peter/Machine_Learning_Practice/model", df_close, scaler, 30
    )

    print(change_r)


if __name__ == "__main__":
    main()
