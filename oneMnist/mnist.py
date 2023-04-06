import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train)
print(x_train[:1].shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(700, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu6),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy'])

his = model.fit(x_train, y_train, epochs=3)

plt.plot(his.history['accuracy'])
plt.plot(his.history['binary_accuracy'])
plt.plot(his.history['categorical_accuracy'])
plt.plot(his.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'binary_accuracy',
           'categorical_accuracy', 'loss'], loc='upper left')
plt.show()

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)

for index, prediction in enumerate(predictions[:25]):
    print(
        f'pred {np.argmax(prediction)} : label {y_test[index]} : right {np.argmax(prediction) == y_test[index]}')

model.evaluate(x_test,  y_test, verbose=2)
