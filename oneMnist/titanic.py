import pandas as pd
import tensorflow as tf
import numpy as np

layers = tf.keras.layers

titanic = pd.read_csv(
    "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

print(titanic.head())

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

inputs = {}
for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


numeric_inputs = {name: input for name,
                  input in inputs.items() if input.dtype == tf.float32}

norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))

x = layers.Concatenate()(list(numeric_inputs.values()))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model=titanic_preprocessing,
                          show_shapes=True, to_file='titanic_preprocessing.png')

titanic_features_dict = {name: np.array(
    value) for name, value in titanic_features.items()}

titanic_sequential = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

preprocessed_inputs2 = titanic_preprocessing(inputs)

result = titanic_sequential(preprocessed_inputs2)

titanic_model = tf.keras.Model(inputs, result)

tf.keras.utils.plot_model(
    model=titanic_model, show_shapes=True, to_file='titanic_model.png')

titanic_model.compile(
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)
