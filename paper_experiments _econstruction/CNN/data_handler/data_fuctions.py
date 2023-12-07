from tensorflow.keras.datasets import mnist
import collections
import utils.config as config
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

TOTAL_NUM_CLIENTS = config.TOTAL_NUM_CLIENTS
NUM_CLIENTS = config.NUM_CLIENTS
BATCH_SIZE =   config.BATCH_SIZE
Input_shape = config.Input_shape

def DataLoaderAndDistributer():

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
  
  return train_dataset,test_dataset

# %%
def preprocess(dataset):
  def batch_format_fn(element):
    return (element['pixels']/255., tf.reshape(element['label'], [1]))
  return dataset.map(batch_format_fn).batch(BATCH_SIZE)

def federated_data_preprocess(train_dataset):

  #prepre dataset for federated model
  # print(train_dataset.client_ids)
  shufled_clients = train_dataset.client_ids
  #random.shuffle(shufled_clients)
  client_ids = shufled_clients[:NUM_CLIENTS]
  # print(client_ids)
  # client_ids = sorted(train_dataset.client_ids)[:NUM_CLIENTS]
  # print(client_ids)
  federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x))
    for x in client_ids
  ]


  return federated_train_data

train_dataset, test_dataset = DataLoaderAndDistributer()
federated_train_data = federated_data_preprocess(train_dataset)
input_spec = federated_train_data[0].element_spec