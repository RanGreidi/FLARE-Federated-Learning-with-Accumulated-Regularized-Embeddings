
import tensorflow as tf
import tensorflow_federated as tff
import data_handler.data_fuctions as data_fuctions
from src.FLARE_regulerizer import FLARE_REGULARIZATION
from src.FedProx_regulerizer import FedProx_REGULARIZATION
import data_handler.data_fuctions as data
#from src.general_utils import *
# %%
def create_keras_model():
    initializer = tf.keras.initializers.GlorotNormal#(seed=0)
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=data_fuctions.Input_shape),
        tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        #tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        #tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        tf.keras.layers.Dense(10, kernel_initializer=initializer),
        tf.keras.layers.Softmax(),
    ])
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
def create_keras_model_for_FedProx(server_weights,tau,u):
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
              new_layer.kernel_regularizer = FedProx_REGULARIZATION(tau,u,server_weights.trainable[2*i])
          if hasattr(new_layer, 'bias_regularizer'):
              #print("acc shape: ",accumolator.trainable[2*i+1].shape)
              #print("layer bias shape: ",new_layer.bias.shape)
              new_layer.bias_regularizer = FedProx_REGULARIZATION(tau,u,server_weights.trainable[2*i+1])
              i +=1
        x = new_layer(x)
      
      reg_model = tf.keras.models.Model(inputs=new_input_layer, outputs=x)
      return reg_model
# %%
def model_fn_for_clients(accumolator,server_weights,tau,u):
    keras_model = create_keras_model_for_FLARE(accumolator,server_weights,tau,u)           
    return tff.learning.models.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])        
def model_fn_for_FedProx(server_weights,tau,u):
    keras_model = create_keras_model_for_FedProx(server_weights,tau,u)           
    return tff.learning.models.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])   
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.models.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
