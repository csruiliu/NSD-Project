
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def model_fn(input_shape, num_classes=10):
  """
      tensorflow.keras sequential x
  """
  x = inputs = Input(shape=input_shape)
  x = Dense(512, activation='relu')(x)
  x = Dense(128, activation='relu')(x)
  outputs = Dense(num_classes, activation='softmax')(x)
  model =  Model(inputs=inputs, outputs=outputs)
  return model

