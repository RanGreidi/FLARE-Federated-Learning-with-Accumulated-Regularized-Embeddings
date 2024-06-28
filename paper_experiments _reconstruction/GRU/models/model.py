import tensorflow as tf
from src.FLARE_regulerizer import FLARE_REGULARIZATION
from tensorflow.keras.models import Sequential

# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
vocab_size = 86

class RNN(tf.keras.models):
  def __init__(self, accumolator,server_weights,tau,u):

    super().__init__(self)
    reg_gru_kernel = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*1],server_weights.trainable[2*1])
    reg_gru_recurrent = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*1],server_weights.trainable[2*1])
    reg_gru_bias = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*1+1],server_weights.trainable[2*1+1])
    reg_dens = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*2],server_weights.trainable[2*2])
    reg_bais = FLARE_REGULARIZATION(tau,u,accumolator.trainable[2*2+1],server_weights.trainable[2*2+1])
    
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    
    self.gru = tf.keras.layers.GRU( rnn_units,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_regularizer=self.reg_gru_kernel,
                                    #recurrent_regularizer=reg_gru,
                                    bias_regularizer=self.reg_gru_bias,
                                )
    
    self.dense = tf.keras.layers.Dense(vocab_size)

    # self.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    # self.add(tf.keras.layers.GRU(   rnn_units,
    #                                 return_sequences=True,
    #                                 return_state=True,
    #                                 #kernel_regularizer=self.reg_gru_kernel,
    #                                 #recurrent_regularizer=reg_gru,
    #                                 #bias_regularizer=self.reg_gru_bias,
    #                             )
    #         )
    # self.add(tf.keras.layers.Dense(vocab_size))
  
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x)

    if return_state:
      return x, states
    else:
      return x