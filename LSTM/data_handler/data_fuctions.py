from tensorflow.keras.datasets import mnist
import collections
import utils.config as config
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from utils.config import vocab,char2idx,idx2char,SEQ_LENGTH,BUFFER_SIZE

TOTAL_NUM_CLIENTS = config.TOTAL_NUM_CLIENTS
NUM_CLIENTS = config.NUM_CLIENTS
BATCH_SIZE =   config.BATCH_SIZE
Input_shape = config.Input_shape


def DataLoaderAndDistributer():

  train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

  # x_train = x_train.astype(np.float32)
  # y_train = y_train.astype(np.int32)
  # x_test = x_test.astype(np.float32)
  # y_test = y_test.astype(np.int32)

  #total_image_count = len(x_train)
  #image_per_set = 600#int(np.floor(total_image_count/TOTAL_NUM_CLIENTS))

  # client_train_dataset = collections.OrderedDict()
  # client_test_dataset = collections.OrderedDict()
  # for i in range(1, TOTAL_NUM_CLIENTS+1):
  #     client_name = "client_" + str(i)
  #     start = image_per_set * (i-1)
  #     end = image_per_set * i

  #     print(f"Adding data from {start} to {end} for client : {client_name}")
  #     data_train = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
  #     client_train_dataset[client_name] = data_train

  #     data_test = collections.OrderedDict((('label', y_test[start:end]), ('pixels', x_test[start:end])))
  #     client_test_dataset[client_name] = data_test

  # train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)
  # test_dataset = tff.simulation.datasets.TestClientData(client_test_dataset)
  
  return train_data,test_data

# %%
# def preprocess(dataset):
#   def batch_format_fn(element):
#     return (element['pixels']/255., tf.reshape(element['label'], [1]))
#   return dataset.map(batch_format_fn).batch(BATCH_SIZE)

def preprocess(dataset):
  return (
      # Map ASCII chars to int64 indexes using the vocab
      dataset.map(to_ids)
      # Split into individual chars
      .unbatch()
      # Form example sequences of SEQ_LENGTH +1
      .batch(SEQ_LENGTH + 1, drop_remainder=True)
      # Shuffle and form minibatches
      .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
      # And finally split into (input, target) tuples,
      # each of length SEQ_LENGTH.
      .map(split_input_target))


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

def generate_text(model, start_string):
  # From https://www.tensorflow.org/tutorials/sequences/text_generation
  num_generate = 200
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(
        predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

def to_ids(x):
  table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)

  s = tf.reshape(x['snippets'], shape=[1])
  chars = tf.strings.bytes_split(s).values
  ids = table.lookup(chars)
  return ids


def split_input_target(chunk):
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


train_dataset, test_dataset = DataLoaderAndDistributer()
federated_train_data = federated_data_preprocess(train_dataset)
input_spec = federated_train_data[0].element_spec