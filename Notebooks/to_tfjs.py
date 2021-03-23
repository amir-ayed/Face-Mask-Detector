import tensorflowjs as tfjs
import tensorflow as tf

tfjs_dir = r'C:\Users\msi\Desktop\h5totfjs\tfjs_model2'

model = tf.keras.models.load_model('mask_detector.model')

tfjs.converters.save_keras_model(model, tfjs_dir)