import tensorflow as tf
import string
import re
import sys

print(sys.argv[1])


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


model = tf.keras.models.load_model('text_classification_model')

print(model.predict(sys.argv[1]))
