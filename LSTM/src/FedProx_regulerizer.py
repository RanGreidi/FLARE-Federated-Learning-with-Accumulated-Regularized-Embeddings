
from tensorflow.keras import regularizers
import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable(package='Custom', name='FedProx_REGULARIZATION')
class FedProx_REGULARIZATION(regularizers.Regularizer):
    def __init__(self, tau,t,server_weights_tensor):
        self.tau = tau
        self.t = t
        self.server_weights = server_weights_tensor
    def __call__(self, w):       
        return (self.tau) * tf.reduce_sum(tf.math.pow(tf.math.abs(tf.subtract( w,self.server_weights)),2)) 
    
    def get_config(self):
        return {'tau': self.tau,
                't': self.t,
                'server_weights': self.server_weights }

