import tensorflow as tf
import tensorflow_federated as tff
from src.general_utils import *

@tf.function
def client_update_EF21(model, dataset, server_weights, prev_difference, client_optimizer, prun_percent, E):
  # Initialize the client model with the current server weights.
  client_weights = tff.learning.ModelWeights.from_model(model)

  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)
  
  # Use the client_optimizer to update the local model.
  for e in range(int(E)):
    for batch in dataset:
      with tf.GradientTape() as tape:
        # Compute a forward pass on the batch of data
        outputs = model.forward_pass(batch)

      # Compute the corresponding gradient
      grads = tape.gradient(outputs.loss, client_weights.trainable)
      grads_and_vars = zip(grads, client_weights.trainable)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads_and_vars)

  #substructe new and old weights
  new_diference_client_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                    client_weights, server_weights)

  #sub prev_difference to the diference_client_weights
  new_diff_minus_pref_diff = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                        new_diference_client_weights, prev_difference)
  
  #create pruned weights diference
  pruned_new_diff_minus_pref_diff = tf.nest.map_structure(lambda x: sparsify_layer(x, prun_percent), 
                                                          new_diff_minus_pref_diff)
  
  
  #add prev diff to new compressed diff
  next_diff = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                                pruned_new_diff_minus_pref_diff, prev_difference)
  
  return pruned_new_diff_minus_pref_diff, next_diff

@tf.function
def server_update_EF21(model, mean_client_diference, server_weights):
  return tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                    mean_client_diference, server_weights)

@tff.tf_computation(model_weights_type, model_weights_type)
def server_update_fn_EF21(weights_difference_mean, server_weights):
  model = model_fn()
  return server_update_EF21(model, weights_difference_mean ,server_weights)

@tff.federated_computation(federated_server_type,federated_server_type)
def server_update_fn_EF21(weights_difference_mean ,server_weights):
  return tff.federated_map(server_update_EF21, (weights_difference_mean ,server_weights))