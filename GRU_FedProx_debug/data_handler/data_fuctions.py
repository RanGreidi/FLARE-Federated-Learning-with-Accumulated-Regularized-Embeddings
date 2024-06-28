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
  return train_data,test_data

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
  client_ids = shufled_clients#[:NUM_CLIENTS]
  # print(client_ids)
  # client_ids = sorted(train_dataset.client_ids)[:NUM_CLIENTS]
  # print(client_ids)
  federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x))
    for x in client_ids
  ]
  new_federated_train_data = []
  for i in range(len(federated_train_data)):
      x=iter(federated_train_data[i])
      j=0
      for k in x:
          j = 1+j
      if j > 30:
        print('number of samples for client ' + str(i) + ':' + str(j)) 
        new_federated_train_data.append(federated_train_data[i])
  new_federated_train_data = new_federated_train_data[:NUM_CLIENTS]
  return new_federated_train_data

def test_data_preprocess(test_dataset, is_test = False):
  shufled_clients = test_dataset.client_ids
  #shufled_clients = shufled_clients[:500]
  def data(client, source=test_dataset):
    return preprocess(source.create_tf_dataset_for_client(client)).take(200)
  test_dataset = tf.data.Dataset.from_tensor_slices([data(client) for client in shufled_clients]).flat_map(lambda x: x)
  return test_dataset

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
central_test = test_data_preprocess(test_dataset)

input_spec = federated_train_data[0].element_spec


# #example input out put
# example_batch = next(iter(federated_train_data[1]))
# example_batch_output = example_batch[1]
# example_batch_inpput = example_batch[0]

# print('output')
# print(idx2char[tf.expand_dims([example_batch_output[2].numpy()], 0) ])
# print('input')
# print(idx2char[tf.expand_dims([example_batch_inpput[2].numpy()], 0) ])