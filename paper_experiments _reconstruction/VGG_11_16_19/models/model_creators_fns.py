
import tensorflow as tf
import data_handler.data_fuctions as data_fuctions
from src.FLARE_regulerizer import FLARE_REGULARIZATION
from models.VGG import VGG19, VGG11, VGG16

def create_keras_model():
  VGG_model = VGG11(data_fuctions.Input_shape)  #Change models here for VGG11 or VGG19
  return VGG_model

def create_keras_model_for_FLARE(accumolator,server_weights,tau,u):
      keras_model = create_keras_model() 
      new_input_layer = tf.keras.layers.Input(shape=data_fuctions.Input_shape)
      x = new_input_layer
      i = 0
      for layer in keras_model.layers:
        config = layer.get_config()
        new_layer = tf.keras.layers.deserialize({'class_name':layer.__class__.__name__,'config': config})

        if new_layer.trainable:
          if hasattr(new_layer, 'kernel_regularizer'):
              #print("acc shape: ",accumolator.trainable[2*i].shape)
              #print("layer kernel shape: ",new_layer.output_shape)          
              new_layer.kernel_regularizer = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*i],server_weights.trainable[2*i])
          if hasattr(new_layer, 'bias_regularizer'):
              #print("acc shape: ",accumolator.trainable[2*i+1].shape)
              #print("layer bias shape: ",new_layer.bias.shape)
              new_layer.bias_regularizer = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*i+1],server_weights.trainable[2*i+1])
              i +=1
        x = new_layer(x)
      
      reg_model = tf.keras.models.Model(inputs=new_input_layer, outputs=x)
      return reg_model

