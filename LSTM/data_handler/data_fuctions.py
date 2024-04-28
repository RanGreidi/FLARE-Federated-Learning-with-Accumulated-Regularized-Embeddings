from tensorflow.keras.datasets import mnist
import collections
import tensorflow.keras as keras
import utils.config as config
import tensorflow as tf
import os
import shutil
import random
import tensorflow_federated as tff
import numpy as np
import string
import re

# get IMDB movie review sentiment data at https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# I followed https://keras.io/examples/nlp/text_classification_from_scratch/

TOTAL_NUM_CLIENTS = config.TOTAL_NUM_CLIENTS
NUM_CLIENTS = config.NUM_CLIENTS
BATCH_SIZE =   config.BATCH_SIZE
Input_shape = config.Input_shape
# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500



def DataLoaderAndDistributer():
  os.umask(0)
  data_dir = "/home/rangrei/Desktop/FLARE/FLARE/LSTM/data_handler/data/aclImdb/train"  
  clients_dir = "/home/rangrei/Desktop/FLARE/FLARE/LSTM/data_handler/data/client_dir" 
  if os.path.exists(clients_dir):
    shutil.rmtree(clients_dir)
  # Create directories for clients
  for i in range(1, NUM_CLIENTS):
      client_dir = os.path.join(clients_dir, f"client_{i}")
      os.makedirs(client_dir, exist_ok=True, mode=0o777)
  # List the files in the 'neg' and 'pos' directories
  neg_files = os.listdir(os.path.join(data_dir, 'neg'))
  pos_files = os.listdir(os.path.join(data_dir, 'pos'))
  # Calculate the number of files per client
  samples_per_set = 600
  neg_files_per_client = 300#len(neg_files) // NUM_CLIENTS
  pos_files_per_client = 300#len(pos_files) // NUM_CLIENTS
  client_train_dataset = collections.OrderedDict()
  client_test_dataset = collections.OrderedDict()
  # Distribute files to clients
  for i in range(0,NUM_CLIENTS-1):
      client_neg_files = neg_files[i * neg_files_per_client: (i + 1) * neg_files_per_client]
      client_pos_files = pos_files[i * pos_files_per_client: (i + 1) * pos_files_per_client]
      client_dir = os.path.join(clients_dir, f"client_{i+1}")
      # Copy 'neg' files to client directory
      for file_name in client_neg_files:
          src_path = os.path.join(data_dir, 'neg', file_name)
          dst_path = os.path.join(client_dir, file_name)
          shutil.copy(src_path, dst_path)
      
      # Copy 'pos' files to client directory
      for file_name in client_pos_files:
          src_path = os.path.join(data_dir, 'pos', file_name)
          dst_path = os.path.join(client_dir, file_name)
          shutil.copy(src_path, dst_path)
      
      client_name = "client_" + str(i)
      print(f"Adding data from {i * samples_per_set} to {samples_per_set + i * samples_per_set} for client : {client_name}")
      
      client_train_dataset[client_name] = # (path_to_txts,labels)

#from here and down it preprocess
  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
      "data_handler/data/aclImdb/train",
      batch_size=None,
      validation_split=0.2,
      subset="training",
      seed=1337,
  )
  raw_test_ds = keras.utils.text_dataset_from_directory(
      "data_handler/data/aclImdb/test",
       batch_size=None
  )

  print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
  print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
  
  # Distributer
  samples_per_set = 600#int(np.floor(total_image_count/TOTAL_NUM_CLIENTS))
  client_train_dataset = collections.OrderedDict()
  client_test_dataset = collections.OrderedDict()
  for i in range(0, TOTAL_NUM_CLIENTS):
    client_name = "client_" + str(i)
    print(f"Adding data from {i * samples_per_set} to {samples_per_set + i * samples_per_set} for client : {client_name}")
    
    data_train = raw_train_ds.skip(i * samples_per_set).take(samples_per_set)
    client_train_dataset[client_name] = data_train.batch(BATCH_SIZE)

    data_test = raw_test_ds.skip(i * samples_per_set).take(samples_per_set)
    client_test_dataset[client_name] = data_test.batch(BATCH_SIZE)

  train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)
  test_dataset = tff.simulation.datasets.TestClientData(client_test_dataset)
  
  # global vectorize_layer
  # vectorize_layer = tf.keras.layers.TextVectorization(
  #                                                         standardize=custom_standardization,
  #                                                         max_tokens=max_features,
  #                                                         output_mode="int",
  #                                                         output_sequence_length=sequence_length,
  #                                                       )
  
  # text_ds = raw_train_ds.map(lambda x, y: x)
  # vectorize_layer.adapt(text_ds)

  # x_train = x_train.astype(np.float32)
  # y_train = y_train.astype(np.int32)
  # x_test = x_test.astype(np.float32)
  # y_test = y_test.astype(np.int32)

  # train_ds = raw_train_ds.map(vectorize_text).unbatch
  # val_ds = raw_val_ds.map(vectorize_text).unbatch
  # test_ds = raw_test_ds.map(vectorize_text).unbatch

  #TODO: move the preprocess to after each client is distibuted with data, first distribute data, than preprocess


  
  return train_dataset,test_dataset

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label



def preprocess(dataset):
  def batch_format_fn(element):
    """Flatten a batch of EMNIST data and return a (features, label) tuple."""
    return (tf.reshape(element['pixels'], [-1, 784]) /255., 
            tf.cast( tf.reshape(element['label'], [-1, 1]), dtype=tf.float32) )

  return dataset.batch(BATCH_SIZE).map(batch_format_fn)

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