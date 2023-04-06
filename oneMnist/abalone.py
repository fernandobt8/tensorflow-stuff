from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)


abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

print(abalone_train.head())

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)

print(abalone_features.shape)

normalizer = layers.Normalization()
normalizer.adapt(abalone_features)

abalone_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(60, activation='relu'),
    layers.Dense(1)
])

tf.keras.utils.plot_model(
    model=abalone_model, show_shapes=True, to_file='abalone_model.png')

abalone_model.compile(loss=tf.losses.MeanAbsoluteError(),
                      optimizer='adam', metrics=['mse', 'mae', 'mape'])

abalone_model.fit(abalone_features, abalone_labels, epochs=100)

predictions = abalone_model.predict(abalone_features[:10])

for index, prediction in enumerate(predictions):
    print(
        f'pred {prediction} : label {abalone_labels[index]}')
