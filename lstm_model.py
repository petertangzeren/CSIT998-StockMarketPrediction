import os
import pickle

import numpy as np
from keras import backend
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


def seperate_data(dataset, time_step=1):
    dataset = dataset.ravel()
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i : (i + time_step)]
        dataX.append(a)
        dataY.append(dataset[i + time_step])
    return np.array(dataX), np.array(dataY)


def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def train(model_path, prices, shift=100):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    x_data, y_data = seperate_data(scaled_prices, shift)
    test_size = int(len(x_data) * 0.3)

    x_test, x_train = x_data[:test_size], x_data[test_size:]
    y_test, y_train = y_data[:test_size], y_data[test_size:]

    model = models.Sequential()

    model.add(layers.LSTM(50, return_sequences=True, input_shape=(shift, 1)))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(1))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=[soft_acc])

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=64,
        verbose=2,
    )
    model.save(model_path)
    scaler_path = os.path.join(model_path, "scaler.pkl")
    loss_acc = model.evaluate(x_test, y_test, verbose=0)
    print("test loss, test acc:", loss_acc)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def predict(model_path, prices, predicting_days=1):
    scaler_path = os.path.join(model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaled_prices = scaler.transform(prices)
    x_data, _ = seperate_data(np.append(scaled_prices, [scaled_prices[-1]]), 100)
    x_input = x_data[-1]

    model = keras.models.load_model(model_path)

    prediction_series = []
    for _ in range(predicting_days):
        x_input = x_input.reshape(1, -1)
        y_predict = model.predict(x_input)
        prediction_series.append(y_predict.ravel())
        x_input = np.append(x_input[0][1:], [y_predict])
    actual_prediction_series = scaler.inverse_transform(prediction_series)
    return actual_prediction_series


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
