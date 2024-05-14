from keras import layers
import tensorflow as tf
import os
import tensorflow_federated as tff
import data_handler.data_fuctions as data_fuctions
from src.FLARE_regulerizer import FLARE_REGULARIZATION
from src.FedProx_regulerizer import FedProx_REGULARIZATION
import data_handler.data_fuctions as data
from utils.config import vocab
#from models.model import RNN
#from src.general_utils import *
# %%
class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

  def __init__(self, name='accuracy', dtype=tf.float32):
    super().__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
    return super().update_state(y_true, y_pred, sample_weight)


def create_keras_model(modify_regularizer = False):
    model = tf.keras.models.clone_model(load_initial_model())
    return model

def create_keras_model_for_FLARE(accumolator,server_weights,tau,u):
      
      def change_layer_regularizer(layer):
        config = layer.get_config()

        # if config['name'] == 'gru_6':
        #   if 'kernel_regularizer' in config:
        #       config['kernel_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2],server_weights.trainable[2])
        #   if 'bias_regularizer' in config:
        #       config['bias_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[3],server_weights.trainable[3])        
        #   if 'bias_regularizer' in config:
        #       config['bias_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[3],server_weights.trainable[3])  

        if config['name'] == 'dense_6':
          if 'kernel_regularizer' in config:
              config['kernel_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[4],server_weights.trainable[4])
          if 'bias_regularizer' in config:
              config['bias_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[5],server_weights.trainable[5])
        
        return layer.__class__.from_config(config)
            
      keras_model = tf.keras.models.clone_model(load_initial_model(),
                                                                      clone_function=lambda layer: change_layer_regularizer(layer))
      
      return keras_model

def create_keras_model_for_FedProx(server_weights,tau,u):
      def change_layer_regularizer(layer):
        config = layer.get_config()

        # if config['name'] == 'gru_6':
        #   if 'kernel_regularizer' in config:
        #       config['kernel_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2],server_weights.trainable[2])
        #   if 'bias_regularizer' in config:
        #       config['bias_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[3],server_weights.trainable[3])        
        #   if 'bias_regularizer' in config:
        #       config['bias_regularizer'] = FLARE_REGULARIZATION(tau,u,accumolator.trainable[3],server_weights.trainable[3])  

        if config['name'] == 'dense_6':
          if 'kernel_regularizer' in config:
              config['kernel_regularizer'] = FedProx_REGULARIZATION(tau,u,server_weights.trainable[4])
          if 'bias_regularizer' in config:
              config['bias_regularizer'] = FedProx_REGULARIZATION(tau,u,server_weights.trainable[5])
        
        return layer.__class__.from_config(config)
            
      keras_model = tf.keras.models.clone_model(load_initial_model(),
                                                                      clone_function=lambda layer: change_layer_regularizer(layer))
      return keras_model

      
# %%
def model_fn_for_clients(accumolator,server_weights,tau,u):
    keras_model = create_keras_model_for_FLARE(accumolator,server_weights,tau,u)           
    return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[FlattenedCategoricalAccuracy()])        
def model_fn_for_FedProx(server_weights,tau,u):
    keras_model = create_keras_model_for_FedProx(server_weights,tau,u)           
    return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[FlattenedCategoricalAccuracy()])   
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[FlattenedCategoricalAccuracy()])

def load_initial_model():
  batch_size = 1
  urls = {
      1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
      8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
  assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
  url = urls[batch_size]
  local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)  
  return tf.keras.models.load_model(local_file, compile=False)


