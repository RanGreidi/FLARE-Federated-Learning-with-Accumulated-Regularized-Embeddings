# %%
import os
import math
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
import numpy as np 
import nest_asyncio
nest_asyncio.apply()
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_federated as tff
from tensorflow import keras
import random
from tensorflow.keras.datasets import mnist
np.random.seed(0)
from contextlib import contextmanager
import sys
from utils.log_writer import log_writer
from utils.plotter import exp_plotter
import utils.config as config

#OUTPUT SUPRESSION
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

#check
tff.federated_computation(lambda: 'Hello, World!')()
# fixed my problem https://stackoverflow.com/questions/63456427/this-version-of-tensorflow-probability-requires-tensorflow-version-2-3 


# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# %%
TOTAL_NUM_CLIENTS = config.TOTAL_NUM_CLIENTS
NUM_CLIENTS = config.NUM_CLIENTS
BATCH_SIZE =   config.BATCH_SIZE
lr = config.lr
MOMENTUM = config.MOMENTUM
K = 1
# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int32)

total_image_count = len(x_train)
image_per_set = 600#int(np.floor(total_image_count/TOTAL_NUM_CLIENTS))

client_train_dataset = collections.OrderedDict()
client_test_dataset = collections.OrderedDict()
for i in range(1, TOTAL_NUM_CLIENTS+1):
    client_name = "client_" + str(i)
    start = image_per_set * (i-1)
    end = image_per_set * i

    print(f"Adding data from {start} to {end} for client : {client_name}")
    data_train = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
    client_train_dataset[client_name] = data_train

    data_test = collections.OrderedDict((('label', y_test[start:end]), ('pixels', x_test[start:end])))
    client_test_dataset[client_name] = data_test

train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)
test_dataset = tff.simulation.datasets.TestClientData(client_test_dataset)

# %%
def preprocess(dataset):
  def batch_format_fn(element):
    """Flatten a batch of EMNIST data and return a (features, label) tuple."""
    return (tf.reshape(element['pixels'], [-1, 784]) /255., 
            tf.cast( tf.reshape(element['label'], [-1, 1]), dtype=tf.float32) )

  return dataset.batch(BATCH_SIZE).map(batch_format_fn)

# %%
#prepre dataset for federated model
print(train_dataset.client_ids)
shufled_clients = train_dataset.client_ids
#random.shuffle(shufled_clients)
client_ids = shufled_clients[:NUM_CLIENTS]
print(client_ids)
# client_ids = sorted(train_dataset.client_ids)[:NUM_CLIENTS]
# print(client_ids)
federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x))
  for x in client_ids
]

#prepre accumolators for clients
def acc_init():
  model = model_fn()
  client_weights = tff.learning.ModelWeights.from_model(model)
  accumolator =   tf.nest.map_structure(lambda x, y: x.assign(tf.zeros(tf.shape(y))),
                                        client_weights, client_weights)
  return [accumolator for i in range(NUM_CLIENTS)]

#acc_init()

# %%
Input_shape = (784,) #taken from the preprosses of the data
def create_keras_model():
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=Input_shape),
        tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        tf.keras.layers.Dense(4069, kernel_initializer=initializer),
        tf.keras.layers.Dense(10, kernel_initializer=initializer),
        tf.keras.layers.Softmax(),
    ])

def create_keras_model_2(accumolator,server_weights,tau,u):
      keras_model = create_keras_model() 
      new_input_layer = tf.keras.layers.Input(shape=Input_shape)
      x = new_input_layer
      i = 0
      for layer in keras_model.layers:
        config = layer.get_config()
        new_layer = tf.keras.layers.deserialize({'class_name':layer.__class__.__name__,'config': config})

        if new_layer.trainable:
          if hasattr(new_layer, 'kernel_regularizer'):
              #print("acc shape: ",accumolator.trainable[2*i].shape)
              #print("layer kernel shape: ",new_layer.output_shape)          
              new_layer.kernel_regularizer = MyRegulariztion(tau,u,accumolator.trainable[2*i],server_weights.trainable[2*i])
          if hasattr(new_layer, 'bias_regularizer'):
              #print("acc shape: ",accumolator.trainable[2*i+1].shape)
              #print("layer bias shape: ",new_layer.bias.shape)
              new_layer.bias_regularizer = MyRegulariztion(tau,u,accumolator.trainable[2*i+1],server_weights.trainable[2*i+1])
              i +=1
        x = new_layer(x)
      
      reg_model = tf.keras.models.Model(inputs=new_input_layer, outputs=x)
      return reg_model


@tf.keras.utils.register_keras_serializable(package='Custom', name='MyRegulariztion')
class MyRegulariztion(regularizers.Regularizer):
    def __init__(self, tau,u, accumolator_tensor,server_weights_tensor):
        self.tau = tau
        self.u = u
        self.accumolator_tensor = accumolator_tensor
        self.server_weights = server_weights_tensor
    def __call__(self, w):       
        #treshhold = tf.math.reduce_mean(tf.math.abs(self.accumolator_tensor))
        treshhold = tfp.stats.percentile(tf.math.abs(self.accumolator_tensor), q=self.u)
        print(treshhold)
        mask =  tf.cast(tf.greater(tf.math.abs(self.accumolator_tensor), treshhold), tf.float32) #those who are the smallest are most updated in acc
        # print(mask)
        return (self.tau) * tf.reduce_sum(tf.math.multiply(mask,tf.math.pow(tf.math.abs(tf.subtract( w,tf.add(self.accumolator_tensor,self.server_weights))),1))) 
    
    def get_config(self):
        return {'tau': self.tau,
                'u': self.u,
                'num_of_calls': self.num_of_calls,
                'accumolator': self.accumolator_tensor,
                'server_weights': self.server_weights }


def model_fn_for_clients(accumolator,server_weights,tau,u):
    keras_model = create_keras_model_2(accumolator,server_weights,tau,u)           
    return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=federated_train_data[0].element_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])        


def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=federated_train_data[0].element_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#path to load a initialzed model
path = os.getcwd()+'/init_model'
def model_fn_for_initiazler():
  keras_model = keras.models.load_model(path)
  return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=federated_train_data[0].element_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# %%
#to save an initialzer model run this
# model_to_save = create_keras_model()
# model_to_save.save('/home/rangrei/Desktop/blbalba_noSeed')

# %%
@tf.function
def prun_layer(layer, prun_percent):
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

# check validity for first layer
#print(tff.learning.ModelWeights.from_model(model_fn())) #print first layer not prunned

pruned = tf.nest.map_structure(lambda x: prun_layer(x, 0.0001),
                            tff.learning.ModelWeights.from_model(model_fn()))
#print(pruned.trainable[0])  #print first layer prunned

#print number of non zero and nu,ber of total weights of all layers
for i in range(len(pruned.trainable)):
  print('number of non zeros weights: ',tf.math.count_nonzero(pruned.trainable[i]).numpy() , '     total weights number in layer:' , pruned.trainable[i].numpy().size  , '     layer shape: ',tf.shape(pruned.trainable[i]).numpy())

# %%
@tf.function
def client_update_my_algo(models, dataset, server_weights, accumolator, client_optimizer, prun_percent, E):
  """Performs training (using the server model weights) on the client's dataset."""
  
  model_0 = models[0]
  # Initialize the client model with the current server weights.
  client_weights_new = tff.learning.ModelWeights.from_model(model_0)
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights_new, server_weights)
  for e in range(K):
    # first epoch
    for batch in dataset:
      with tf.GradientTape() as tape:
        # Compute a forward pass on the batch of data
        outputs = model_0.forward_pass(batch)

      # Compute the corresponding gradient
      grads = tape.gradient(outputs.loss, client_weights_new.trainable)
      grads_and_vars = zip(grads, client_weights_new.trainable)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads_and_vars)
  
  client_weights_old = client_weights_new


  model_1 = models[1]
  client_weights_new = tff.learning.ModelWeights.from_model(model_1)
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights_new, client_weights_old)  

  for e in range(int(E-K)):      
      # second epoch
      for batch in dataset:
        with tf.GradientTape() as tape:
          # Compute a forward pass on the batch of data
          outputs = model_1.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights_new.trainable)
        grads_and_vars = zip(grads, client_weights_new.trainable)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)


  #substructe new and old weights
  diference_client_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                    client_weights_new, server_weights)

  #add accumolator to the diference_client_weights
  diff_plus_acc = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                        diference_client_weights, accumolator)
  
  #create pruned weights diference
  pruned_client_diference_weights = tf.nest.map_structure(lambda x: prun_layer(x, prun_percent), 
                                                          diff_plus_acc)
  
  #create inverse pruned weights diference (by substructe the pruned from the not pruned)
  inverse_pruned_client_diference_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                                diff_plus_acc, pruned_client_diference_weights)
  
  #assign the  inverse pruned weights diference to the accumolator
  accumolator = inverse_pruned_client_diference_weights
  
  return pruned_client_diference_weights, accumolator

# %%
@tf.function
def client_update(model, dataset, server_weights, accumolator, client_optimizer, prun_percent, E):
  """Performs training (using the server model weights) on the client's dataset."""
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
  diference_client_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                    client_weights, server_weights)

  #add accumolator to the diference_client_weights
  diff_plus_acc = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                        diference_client_weights, accumolator)
  
  #create pruned weights diference
  pruned_client_diference_weights = tf.nest.map_structure(lambda x: prun_layer(x, prun_percent), 
                                                          diff_plus_acc)
  
  #create inverse pruned weights diference (by substructe the pruned from the not pruned)
  inverse_pruned_client_diference_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                                diff_plus_acc, pruned_client_diference_weights)
  
  #assign the  inverse pruned weights diference to the accumolator
  accumolator = inverse_pruned_client_diference_weights
  
  return pruned_client_diference_weights, accumolator

# %%
@tf.function
def FedAvg_client_update(model, dataset, server_weights, client_optimizer, E):
  """Performs training (using the server model weights) on the client's dataset."""
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
  return client_weights

# %%
@tff.tf_computation
def server_init():
  model = model_fn_for_initiazler()
  return tff.learning.ModelWeights.from_model(model)

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

# %%
# tf.computations types
whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
print(str(tf_dataset_type))

model_weights_type = server_init.type_signature.result
print(str(model_weights_type))

prun_percent_type = tf.constant(1, dtype = tf.float32).dtype
print(str(prun_percent_type))


#federated types
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
#print(federated_server_type)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
#print(federated_dataset_type)
federated_clients_type = tff.FederatedType(model_weights_type, tff.CLIENTS)
#print(federated_clients_type)
federated_prun_percent_type = tff.FederatedType(prun_percent_type, tff.CLIENTS)
#print(federated_prun_percent_type)


# %%
@tf.function
def server_update(model, mean_client_diference, server_weights):
  ''' 
  input: (tf.model - model like meam diference, tf.model.trainable_variables - mean_client_diference , tf.model.trainable_variables - server_state)
  output: tf.model.trainable_variables - with the sum of deference and the server state model
  '''
  return tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                    mean_client_diference, server_weights)
  

# #debug
# model_deb1 = create_keras_model()
# weights_deb1 = model_deb1.trainable_variables
# print(weights_deb1,"\n")

# model_deb2 = create_keras_model()
# weights_deb2 = model_deb2.trainable_variables
# print(weights_deb2,"\n")

# mode = create_keras_model()
# l = server_update(mode, weights_deb1, weights_deb2)
# print(l)


# %%
@tff.tf_computation(model_weights_type, model_weights_type)
def server_update_fn(weights_difference_mean, server_weights):
  model = model_fn()
  return server_update(model, weights_difference_mean ,server_weights)

@tff.federated_computation(federated_server_type,federated_server_type)
def server_update_fn(weights_difference_mean ,server_weights):
  return tff.federated_map(server_update_fn, (weights_difference_mean ,server_weights))
 

# %%
#My algo

@tff.tf_computation(tf_dataset_type, model_weights_type, model_weights_type, prun_percent_type, prun_percent_type, prun_percent_type, prun_percent_type,prun_percent_type)
def client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u):
  # models_0 = model_fn_for_clients(accumolator,server_weights,tf.math.scalar_mul(0,tau),u)
  # models_1 = model_fn_for_clients(accumolator,server_weights,tf.math.scalar_mul(0,tau),u)
  # models_2 = model_fn_for_clients(accumolator,server_weights,tf.math.scalar_mul(0,tau),u)
  # models_3 = model_fn_for_clients(accumolator,server_weights,tf.math.scalar_mul(0,tau),u)
  # models_4 = model_fn_for_clients(accumolator,server_weights,tf.math.scalar_mul(0,tau),u)
  models_0 = model_fn_for_clients(accumolator,server_weights,tau,u)
  models_1 = model_fn()
  models_2 = model_fn()
  models_3 = model_fn()
  models_4 = model_fn()
  models = [models_0,models_1,models_2,models_3,models_4]
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=MOMENTUM)
  pruned_client_weights, accumolator = client_update_my_algo(models, tf_dataset, server_weights, accumolator, client_optimizer, prun_percent, E) 
  return pruned_client_weights, accumolator
 
@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_clients_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type,federated_prun_percent_type,federated_prun_percent_type)
def client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u):
  return tff.federated_map(client_update_fn, (tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u))


# %%
# #debuggg
# initial_server_state = initialize_fn()
# accumolators = acc_init()
# federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x))
#   for x in client_ids
# ]

# ss = [initial_server_state for i in range (NUM_CLIENTS)]

# learning_rate  = [lr for i in range (NUM_CLIENTS)]
# tau = [0.1 for i in range (NUM_CLIENTS)]  
# u = [50 for i in range (NUM_CLIENTS)] 
# prun_percent = [10 for i in range(NUM_CLIENTS)]     
# cleints_E_algo = [1 for i in range(NUM_CLIENTS)] 


# x = client_update_fn(federated_train_data, ss, accumolators, prun_percent, learning_rate, cleints_E_algo, tau,u)

# %%
#Second Algo

@tff.tf_computation(tf_dataset_type, model_weights_type, model_weights_type, prun_percent_type, prun_percent_type, prun_percent_type)
def Second_algo_client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E):
  #model = model_fn_for_clients(accumolator,server_weights,tau,u)  
  model = model_fn() #build regular model
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=MOMENTUM, decay=0.01)
  pruned_client_weights, accumolator = client_update(model, tf_dataset, server_weights, accumolator, client_optimizer, prun_percent, E) 
  return pruned_client_weights, accumolator
 
@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_clients_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type)
def Second_algo_client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E):
  return tff.federated_map(Second_algo_client_update_fn, (tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E))

# %%
#fed avg
@tff.tf_computation(tf_dataset_type, model_weights_type, prun_percent_type)
def FedAvg_client_update_fn(tf_dataset, server_weights, E):
  model = model_fn()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=MOMENTUM, decay=0.01)
  client_weights = FedAvg_client_update(model, tf_dataset, server_weights, client_optimizer, E)
  return client_weights

@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_prun_percent_type)
def FedAvg_client_update_fn(tf_dataset, server_weights, E):
  return tff.federated_map(FedAvg_client_update_fn, (tf_dataset, server_weights, E))


# %%
@tf.function
def FedAvg_server_update(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  return mean_client_weights

@tff.tf_computation(model_weights_type)
def FedAvg_server_updatefn(mean_client_weights):
  model = model_fn()
  return FedAvg_server_update(model, mean_client_weights)

@tff.federated_computation(federated_server_type)
def FedAvg_server_update_fn(server_weights):
  return tff.federated_map(FedAvg_server_updatefn, server_weights)

# %%
#My algo

@tff.federated_computation(federated_server_type, federated_clients_type, federated_dataset_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type,federated_prun_percent_type)
def next_fn(server_weights, accumoltors, federated_dataset, prun_percent, learning_rate, E, tau,u):
  # Broadcast the server weights to the clients. server -> clients
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights. clients -> clients
  pruned_client_weights_diference, accumoltors = client_update_fn(federated_dataset, server_weights_at_client, accumoltors, prun_percent, learning_rate, E, tau,u)

  # The server averages these weights_difference. clients -> server
  weights_difference_mean = tff.federated_mean(pruned_client_weights_diference)

  #the server adds the old weights to weights_difference and updates its model. server -> server
  server_weights = server_update_fn(weights_difference_mean ,server_weights)

  return server_weights, accumoltors

str(next_fn.type_signature)


# %%
#Second algo 

@tff.federated_computation(federated_server_type, federated_clients_type, federated_dataset_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type)
def Second_algo_next_fn(server_weights, accumoltors, federated_dataset, prun_percent, learning_rate, E):
  # Broadcast the server weights to the clients. server -> clients
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights. clients -> clients
  pruned_client_weights_diference, accumoltors = Second_algo_client_update_fn(federated_dataset, server_weights_at_client, accumoltors, prun_percent, learning_rate,E)

  # The server averages these weights_difference. clients -> server
  weights_difference_mean = tff.federated_mean(pruned_client_weights_diference)

  #the server adds the old weights to weights_difference and updates its model. server -> server
  server_weights = server_update_fn(weights_difference_mean ,server_weights)

  return server_weights, accumoltors


# %%
@tff.federated_computation(federated_server_type, federated_dataset_type,federated_prun_percent_type)
def FedAvg_next_fn(server_weights, federated_dataset, E):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = FedAvg_client_update_fn(federated_dataset, server_weights_at_client, E)

  # The server averages these updates.
  #mean_client_weights = tff.federated_mean(pruned_client_weights)
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = FedAvg_server_update_fn(mean_client_weights)

  return server_weights
str(FedAvg_next_fn.type_signature)

# %%
central_emnist_test = test_dataset.create_tf_dataset_from_all_clients()
central_emnist_test = preprocess(central_emnist_test)

def evaluate(server_state):
  keras_model = create_keras_model()
  keras_model.compile(
      #loss=Loss_Fn(keras_model.losses),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  server_state.assign_weights_to(keras_model)
  res = keras_model.evaluate(central_emnist_test)
  return res

# evaluate initialzed model 
server_state = initialize_fn()
history = evaluate(server_state)
print(server_state)


# %%
begin_acc_realse = config.begin_acc_realse
end_acc_realse = config.end_acc_realse
update_acc_realse = config.update_acc_realse
max_prec_to_prun = config.max_prec_to_prun
max_E = config.max_E
m_treshhold = config.m_treshhold
early_stop = config.early_stop


def calc_multypliers(round,server_state_algo,prun_percent_multyplier,learning_rate_multyplier, E_adder,PRUN_PERCENT,E):
        with suppress_stdout():
            eval = evaluate(server_state_algo)            
            current_loss = eval[0]
            current_acc = eval[1]
        #print('loss My',current_loss,'acc My',  current_acc)    
        if round == 0 :
          global acc_memory
          acc_memory = []
          acc_memory.append(current_acc)
        elif round <= update_acc_realse:
          acc_memory.append(current_acc)
        else:
          acc_memory.append(current_acc)
          acc_memory.pop(0)   
        
        if (round%update_acc_realse == 0) and (round > begin_acc_realse-1) and (round < end_acc_realse):
            #m = ((current_acc)-(acc_memory[0]))/(current_acc)
            m = abs(((current_acc)-(acc_memory[0])))           
            #print('m', m)
            learning_rate_multyplier = 1
            if  m < m_treshhold:
              if prun_percent_multyplier*PRUN_PERCENT < max_prec_to_prun:
                prun_percent_multyplier = prun_percent_multyplier*config.prun_percent_multyplier_const
              if E_adder+E > E/2:
                E_adder = E_adder-config.E_multyplier_const           
              return prun_percent_multyplier, learning_rate_multyplier, E_adder
            else:
              if prun_percent_multyplier != 1:
                  prun_percent_multyplier = prun_percent_multyplier/config.prun_percent_multyplier_const
              if E_adder+E < E:
                  E_adder  = E_adder+config.E_multyplier_const 
              return prun_percent_multyplier, learning_rate_multyplier, E_adder
        elif current_acc > early_stop:
            prun_percent_multyplier = 1
            E_adder = 0#1
            learning_rate_multyplier = 1          
            return prun_percent_multyplier, learning_rate_multyplier, E_adder
        else :                            
            return prun_percent_multyplier, learning_rate_multyplier, E_adder

# %%
def calc_multypliers_FFL(history_federeted,second_algo_server_state,PRUN_PERCENT,E):       
        with suppress_stdout():
                eval = evaluate(second_algo_server_state)
        current_loss = eval[0]
        #print('loss FFL ',eval[0], 'acc FFL', eval[1])
        First_loss = np.array(history_federeted)[:,0][0]
        prun_percent_FFL = PRUN_PERCENT * np.power((First_loss/current_loss),1/3)
        E_FFL = np.round(E * np.power((current_loss/First_loss),1/3))
        if E_FFL < 1:
                E_FFL = 1  
        #print(current_loss) 
        #print(First_loss)
                               
        return prun_percent_FFL, E_FFL

# %%
initial_server_state = initialize_fn()

ploter_dic = {}
exp_name_list = []
num_of_experiments = 1
experiments = {              
              'ROUNDS':               [1001,1001,1001,1001,1001,1001,1001,1001,201],
              'PRUN_PERCENT' :        [0.001,0.001,0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.0001],
              'E':                    [1,8,16,32,1,4,8,16,1],
              'TAU':                  [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],                  
              'tau_decay_const':      [1.05,1.05,1.05,1.05,1.05,1.005,1.005,1.005,0],
              'u':                    [50,50,50,50,50,50,50,50,50]
              }

for experiment in range(num_of_experiments):
    
    #prepare experiment
    PRUN_PERCENT = experiments['PRUN_PERCENT'][experiment]
    ROUNDS = experiments['ROUNDS'][experiment]
    E = experiments['E'][experiment]
    TAU = experiments['TAU'][experiment]
    TAU_DECAY_CONST = experiments['tau_decay_const'][experiment]
    U = experiments['u'][experiment]

    experiment_name = 'FC_{}R_{}E_{}TAU_{}CLIENTS_{}ROUNDS_{}Decay_{}MaxP_{}u_m{}OSR_{}RegSteps'.format(PRUN_PERCENT,E,TAU,NUM_CLIENTS,ROUNDS,TAU_DECAY_CONST,config.max_prec_to_prun,U,config.m_treshhold,K)
    exp_name_list.append(experiment_name)
    
    #make dir for experiment
    if not os.path.exists('results/' + experiment_name): 
        os.makedirs('results/' + experiment_name)

    #initialize expiriment
    history_federeted = []
    history_FedAvg = []
    history_second_algo_server_state = []
    
    output_list = []
    prun_precent_logger = []
    E_logger = []
    prun_precent_logger_FFL = []
    E_logger_FFL = []

    server_state = initial_server_state
    accumolators = acc_init()
    accumolators_second_algo = acc_init()

    server_state_FedAvg = server_state
    server_state_algo = server_state
    second_algo_server_state = server_state

    prun_percent_multyplier = 1
    learning_rate_multyplier = 1
    E_adder = 0
    tau_decay = 1

    for round in range(ROUNDS):
      #evaluate every 10 rounds
      print("round:",round)
      if round % 10 == 0 :           
          print('FederatedAlgo evaluation')
          a = evaluate(server_state_algo)
          history_federeted.append(a)            
          print('FedAvg evaluation')
          b = evaluate(server_state_FedAvg)
          history_FedAvg.append(b)  
          print('Second Algo evaluation')
          c = evaluate(second_algo_server_state)
          history_second_algo_server_state.append(c)

          log_writer(experiment_name,
                      history_federeted,
                      history_FedAvg,
                      history_second_algo_server_state,
                      prun_precent_logger,
                      E_logger,
                      prun_precent_logger_FFL,
                      E_logger_FFL,
                      output_list, a,b,c,round)

      # prun_percent_multyplier, learning_rate_multyplier, E_adder =  calc_multypliers(round,
      #                                                                               server_state_algo,
      #                                                                            prun_percent_multyplier,
      #                                                                            learning_rate_multyplier,
      #                                                                          E_adder,
      #                                                                          PRUN_PERCENT,
      #                                                                         E)           
      prun_percent_FFL, E_FFL =  calc_multypliers_FFL(history_federeted,
                                                      second_algo_server_state,
                                                      PRUN_PERCENT,
                                                      E)                       
      tau_decay = tau_decay*TAU_DECAY_CONST
      

      #for my algo
      learning_rate  = [lr for i in range(NUM_CLIENTS)]
      tau = [TAU/tau_decay for i in range(NUM_CLIENTS)]  
      u = [U for i in range (NUM_CLIENTS)] 
      prun_percent = [PRUN_PERCENT for i in range(NUM_CLIENTS)]                   #prun_percent = [PRUN_PERCENT*prun_percent_multyplier for i in range(NUM_CLIENTS)]     
      cleints_E_algo = [np.round(E) for i in range(NUM_CLIENTS)]                             #cleints_E_algo = [E + E_adder for i in range(NUM_CLIENTS)]      
      #for second algo
      prun_percent_second_algo = [prun_percent_FFL for i in range(NUM_CLIENTS)]
      cleints_E_second_algo = [E_FFL for i in range(NUM_CLIENTS)]
      #for fedavg
      cleints_E = [E for i in range(NUM_CLIENTS)]

      prun_precent_logger.append(prun_percent[0])    
      E_logger.append(cleints_E_algo[0])                            
      prun_precent_logger_FFL.append(prun_percent_second_algo[0])
      E_logger_FFL.append(cleints_E_second_algo[0])      

      # #print parameters
      #print(f'R_FFL {prun_percent_FFL}')
      #print(f'E_FFL {E_FFL}') 
      #print(f'R_My {prun_percent_multyplier*PRUN_PERCENT}')
      #print(f'E_My {E_multyplier+E}')  
      #print(f'lr_My {lr*learning_rate_multyplier}')
      #print(f'tau {tau[0]}')
      #print(f'u {u[0]}')
      
      # #Choose cleints
      # shufled_clients = train_dataset.client_ids
      # random.shuffle(shufled_clients)
      # client_ids = shufled_clients[:NUM_CLIENTS]
      # federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x)) for x in client_ids]      

      #algorithmed federeted training in the round
      server_state_algo, accumolators = next_fn(server_state_algo, accumolators, federated_train_data, prun_percent, learning_rate, cleints_E_algo, tau,u)      
      #conventional federated training in the round
      server_state_FedAvg = FedAvg_next_fn(server_state_FedAvg, federated_train_data, cleints_E)       
      #Other Algorithem
      second_algo_server_state, accumolators_second_algo = Second_algo_next_fn(second_algo_server_state, accumolators_second_algo, federated_train_data, prun_percent_second_algo, learning_rate, cleints_E_second_algo)

    #----------------- plot experiment -----------------------
    exp_plotter(history_federeted,
                history_FedAvg,
                history_second_algo_server_state,
                prun_precent_logger,
                prun_precent_logger_FFL,
                E_logger,
                E_logger_FFL,
                ROUNDS,
                experiment_name)

# %%
# measure is number how much non zeros weights we have in out methid

# pruned = tf.nest.map_structure(lambda x: prun_layer(x, 0.001),
#                             server_state_algo)
# #print(pruned.trainable[0])  #print first layer prunned
# #print number of non zero and nu,ber of total weights of all layers
# for i in range(len(pruned.trainable)):
  
#   print('number of non zeros weights: ',tf.math.count_nonzero(pruned.trainable[i]).numpy() , '     total weights number in layer:' , pruned.trainable[i].numpy().size  , '     layer shape: ',tf.shape(pruned.trainable[i]).numpy())

# pruned.trainable[1]

# %%
# plt.figure(figsize=(14,6))
# plt.title("title") 
# plt.xlabel("rounds") 
# plt.plot(ploter_dic[exp_name_list[0]]['MyAlgo'] , '-', color='green' ,label=exp_name_list[0]+' MyAlgo') 
# plt.plot(ploter_dic[exp_name_list[0]]['Second_algo'] , '--', color='green' , label=exp_name_list[0]+' Second_algo')

# plt.plot(ploter_dic[exp_name_list[1]]['MyAlgo'] , '-', color='orange' ,label=exp_name_list[1]+' MyAlgo') 
# plt.plot(ploter_dic[exp_name_list[1]]['Second_algo'] , '--', color='orange' , label=exp_name_list[1]+' Second_algo')

# plt.plot(ploter_dic[exp_name_list[2]]['MyAlgo'] , '-', color='blue' ,label=exp_name_list[2]+' MyAlgo') 
# plt.plot(ploter_dic[exp_name_list[2]]['Second_algo'] , '--', color='blue' ,label=exp_name_list[2]+' Second_algo')

# # plt.plot(ploter_dic[exp_name_list[3]]['MyAlgo'] , '-', color='red' ,label=exp_name_list[3]+' MyAlgo') 
# # plt.plot(ploter_dic[exp_name_list[3]]['Second_algo'] , '--', color='red' ,label=exp_name_list[3]+' Second_algo')
# plt.legend()
# #plt.savefig('prun_precent_logger.png', bbox_inches='tight')

# plt.savefig('exp4_enlarging_E', bbox_inches='tight')
# plt.show()




