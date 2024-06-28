import tensorflow as tf
import tensorflow_federated as tff
from models.model_creators_fns import model_fn,model_fn_for_initiazler

@tf.function
def sparsify_layer(layer, prun_percent):
  #current bug: canot excid 85%
  #input:   tff.learning.ModelWeights.from_model(model)
  #output:  tff.learning.ModelWeights.from_model(model)
    precent_to_zero = prun_percent # as %, its the precentege of the element to keep, if equal to 10, than only 10 largest will remain
    flat_layer =  tf.reshape(layer,[-1])
    k = tf.get_static_value(tf.size(flat_layer)) * (precent_to_zero/100)   # k is the number of eleement that is the top k
    b = tf.nn.top_k(tf.abs(flat_layer), tf.cast(tf.round(k)+2,tf.int32))
    kth = tf.reduce_min(b.values)
    #print(kth)
    mask = tf.greater(tf.abs(layer), kth * tf.ones_like(layer))
    prunned_layer = tf.multiply(layer, tf.cast(mask, tf.float32))
    return prunned_layer
@tff.tf_computation

def server_init():
  model = model_fn()
  return tff.learning.ModelWeights.from_model(model)

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

@tf.function
def server_update(model, mean_client_diference, server_weights):
  return tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                    mean_client_diference, server_weights)

#types defenition
whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
model_weights_type = server_init.type_signature.result
prun_percent_type = tf.constant(1, dtype = tf.float32).dtype

#federated types
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
federated_clients_type = tff.FederatedType(model_weights_type, tff.CLIENTS)
federated_prun_percent_type = tff.FederatedType(prun_percent_type, tff.CLIENTS)