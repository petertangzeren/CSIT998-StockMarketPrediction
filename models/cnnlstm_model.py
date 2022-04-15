import os
import pickle

import numpy as np
from keras import backend
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


def separate_data(dataset, time_step=1, predicting_steps_after=1):
    dataset = dataset.ravel()
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - predicting_steps_after + 1):
        a = dataset[i : (i + time_step)]
        dataX.append(a)
        dataY.append(dataset[i + time_step + predicting_steps_after - 1])
    return np.array(dataX), np.array(dataY)


def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def train(model_path, prices, shift=100, predicting_days=5):
    if prices.empty:
        return
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    x_data, y_data = separate_data(scaled_prices, shift, predicting_days)
    test_size = int(len(x_data) * 0.2)

    x_test, x_train = x_data[:test_size], x_data[test_size:]
    y_test, y_train = y_data[:test_size], y_data[test_size:]

    model = models.Sequential()

    model.add(layers.LSTM(50, return_sequences=True, input_shape=(shift, 1)))
    model.add(layers.Conv1D(50, kernel_size=3))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(model.summary())

    training = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        verbose=2,
    )
    model.save(model_path)
    scaler_path = os.path.join(model_path, "scaler.pkl")
    loss_acc = model.evaluate(x_test, y_test, verbose=0)
    print("test loss, test acc:", loss_acc)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    plt.plot(training.history["loss"])
    plt.plot(training.history["val_loss"])
    plt.show()


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


def main():
    import pandas_datareader as pdr

    ticker = "MSFT"
    api_key = "aeeaa9dbc8f82f2c361abaa259050d75e736b424"
    price_history = pdr.get_data_tiingo(ticker, api_key=api_key)
    prices = price_history[["close"]].pct_change().dropna()
    model_path = f"lstm_models/{ticker}_close"
    train(model_path, prices)
    prediction = predict(model_path, prices, predicting_days=5)
    print(prediction)


if __name__ == "__main__":
    main()
