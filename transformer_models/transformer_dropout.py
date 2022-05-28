import numpy as np
from keras import backend
import pickle
import pandas as pd
import os, datetime
from tensorflow import keras
import tensorflow as tf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers

print("Tensorflow version: {}".format(tf.__version__))

import warnings

warnings.filterwarnings("ignore")

batch_size = 32
seq_len = 100

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

predict_days = 5


def train(model_path, df):
    scaler_1 = StandardScaler()
    close_data = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    scaled_close = scaler_1.fit_transform(close_data)

    num_train_samples = int(0.5 * len(scaled_close))
    num_val_samples = int(0.25 * len(scaled_close))
    num_test_samples = len(scaled_close) - num_train_samples - num_val_samples
    print("num_train_samples:", num_train_samples)
    print("num_val_samples:", num_val_samples)
    print("num_test_samples:", num_test_samples)

    def convert_to_dataset(array):
        delay = predict_days
        targets = (
            pd.DataFrame(array).rolling(window=5).apply(lambda x: np.prod(1 + x) - 1)
        )
        targets = targets[delay:]
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            array[:-delay],
            targets=targets,
            sampling_rate=1,
            sequence_length=200,
            shuffle=False,
            batch_size=200,
            start_index=None,
            end_index=None,
        )
        return dataset

    train_array = scaled_close[:num_train_samples]
    train_dataset = convert_to_dataset(train_array)

    validation_array = scaled_close[
        num_train_samples : num_train_samples + num_val_samples
    ]
    val_dataset = convert_to_dataset(validation_array)

    test_array = scaled_close[num_train_samples + num_val_samples :]
    test_dataset = convert_to_dataset(test_array)

    input_shape = (
        200,
        scaled_close.shape[-1],
    )

    # https://keras.io/examples/timeseries/timeseries_transformer_classification/
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    # https://keras.io/examples/timeseries/timeseries_classification_transformer/
    def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

    model = build_model(
        input_shape,
        head_size=50,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"{model_path}/jena_stacked_transformer_dropout.keras", save_best_only=True
        )
    ]
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=callbacks,
    )
    model = keras.models.load_model(
        f"{model_path}/jena_stacked_transformer_dropout.keras"
    )
    print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
    input_for_prediction = scaled_close[-200:].reshape(1, 200, 1)
    prediction = model.predict(input_for_prediction)
    reversed_prediction = scaler_1.inverse_transform(prediction).ravel()[0]

    print(f"Prediction: Close after 5 days would be {reversed_prediction:.2f}")

    model.save(model_path)
    scaler_path = os.path.join(model_path, "scaler.pkl")
    loss_acc = model.evaluate(test_dataset, verbose=0)
    print("test loss, test acc:", loss_acc)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_1, f)
    # import matplotlib.pyplot as plt
    # plt.style.use("seaborn")
    # plt.plot(history.history["loss"])
    # plt.plot(history.history["val_loss"])
    # plt.show()

    return model, scaler_1


def predict(model_path, prices, predicting_days=5):
    scaler_path = os.path.join(model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaled_prices = scaler.transform(prices)
    x_input = scaled_prices[-200:].T

    model = keras.models.load_model(model_path)
    y_predict = model.predict(x_input)
    future = scaler.inverse_transform(y_predict).ravel()[0]
    daily_prediction = (1 + future) ** (1 / predicting_days) - 1

    prediction_series = [daily_prediction] * predicting_days
    return np.array(prediction_series)


def main():
    ticker = "MSFT"
    df = pdr.get_data_tiingo(ticker, api_key="aeeaa9dbc8f82f2c361abaa259050d75e736b424")
    df.to_csv("MSFT.csv")
    df = pd.read_csv("MSFT.csv", delimiter=",", usecols=["date", "close"])
    df.sort_values("date", inplace=True)
    df = df[["close"]].pct_change().dropna()

    model_path = f"transformer_models/{ticker}_close_transformer"

    train(model_path, df)
    prediction = predict(model_path, df, predicting_days=predict_days)
    print(prediction)


if __name__ == "__main__":
    main()
