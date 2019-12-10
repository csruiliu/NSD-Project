
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications import ResNet50

def model_fn(input_shape, num_classes=10):
  """
      tensorflow.keras sequential x
  """

  x = inputs = Input(shape=input_shape)
  base_model =  ResNet50(False,None,x, classes=num_classes)
  x = base_model.output
  x = Flatten()(x)
  x = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs,x)
  return model
