import tensorflow as tf
import tensorflow_federated as tff
from src.general_utils import *

@tf.function
def client_update_EF21(model, dataset, server_weights, prev_state, client_optimizer, prun_percent, E):
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
                                        new_diference_client_weights, prev_state)
  
  #create pruned weights diference
  pruned_new_diff_minus_pref_diff = tf.nest.map_structure(lambda x: sparsify_layer(x, prun_percent), 
                                                          new_diff_minus_pref_diff)
  
  #add prev diff to new compressed diff
  next_state = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                                pruned_new_diff_minus_pref_diff, prev_state)

  #cheet
  pruned_new_diff_minus_pref_diff_cheet = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                                                pruned_new_diff_minus_pref_diff, prev_state) 

  return pruned_new_diff_minus_pref_diff_cheet, next_state

@tf.function
def server_update_EF21(model, mean_client_diference, server_weights):
  return tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                    mean_client_diference, server_weights)
