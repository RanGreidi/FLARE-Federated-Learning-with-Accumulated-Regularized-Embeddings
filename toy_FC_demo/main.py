# %%
import os
from matplotlib import pyplot as plt
import numpy as np 
import nest_asyncio
nest_asyncio.apply()
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
#np.random.seed(0)
from utils.log_writer import log_writer
from utils.plotter import exp_plotter
import utils.config as config
from data_handler.data_fuctions import *
from models.model_creators_fns import *
from src.FLARE_utils import *

tff.federated_computation(lambda: 'Hello, World!')()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
train_dataset, test_dataset = DataLoaderAndDistributer()
federated_train_data = federated_data_preprocess(train_dataset)
central_test = test_dataset.create_tf_dataset_from_all_clients()
central_test = preprocess(central_test)

# %%
#print numbers of non zero and numbers of total weights of all layers according to predefined R
R = 0.001
pruned = tf.nest.map_structure(lambda x: sparsify_layer(x, R),
                            tff.learning.ModelWeights.from_model(model_fn()))
for i in range(len(pruned.trainable)):
  print('number of non zeros weights: ',tf.math.count_nonzero(pruned.trainable[i]).numpy() , '     total weights number in layer:' , pruned.trainable[i].numpy().size  , '     layer shape: ',tf.shape(pruned.trainable[i]).numpy())

# %%
initial_server_state = initialize_fn()

ploter_dic = {}
exp_name_list = []
num_of_experiments = 1
NUM_CLIENTS = config.NUM_CLIENTS

experiments = {              
              'ROUNDS':               [1001,1001,1001,1001,1001,1001,1001,1001,201],
              'R' :                   [0.001,0.001,0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.0001],
              'E':                    [1,8,16,32,1,4,8,16,1],
              'TAU':                  [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],                  
              'c':                    [1.05,1.05,1.05,1.05,1.05,1.005,1.005,1.005,0],
              't':                    [50,50,50,50,50,50,50,50,50]
              }

for experiment in range(num_of_experiments):
    
    #prepare experiment
    R = experiments['R'][experiment]
    ROUNDS = experiments['ROUNDS'][experiment]
    E = experiments['E'][experiment]
    TAU = experiments['TAU'][experiment]
    c = experiments['c'][experiment]
    t = experiments['t'][experiment]

    experiment_name = '{}R_{}E_{}TAU_{}CLIENTS_{}ROUNDS_{}Decay_{}t_{}p'.format(R,E,TAU,NUM_CLIENTS,ROUNDS,c,t,config.p)
    exp_name_list.append(experiment_name)
    
    #make dir for experiment results
    if not os.path.exists('results/' + experiment_name): 
        os.makedirs('results/' + experiment_name)

    #initialize expiriment
    history_FLARE = []
    history_FedAvg = []
    history_FedProx_server_state = []
    history_EF21_server_state = []
    history_EC_server_state = []

    output_list = []
    prun_precent_logger_FLARE = []
    E_logger_FLARE = []
    prun_precent_logger_FFL = []
    E_logger_FFL = []
    prun_precent_logger_EF21 = []
    E_logger_EF21 = []
    prun_precent_logger_EC = []
    E_logger_EC = []

    server_state = initial_server_state
    accumolators_FLARE = acc_init()
    accumolators_FedProx = acc_init()
    accumolators_EF21 = acc_init()
    accumolators_EC = acc_init()

    server_state_FedAvg = server_state
    server_state_FLARE = server_state
    server_state_FedProx = server_state
    server_state_EF21 = server_state
    server_state_EC = server_state

    tau_decay = 1

    for round in range(ROUNDS):
      #evaluate every 10 rounds
      print("round:",round)
      if round % 10 == 0 :           
          print('FLARE evaluation')
          A = evaluate(server_state_FLARE, central_test)
          history_FLARE.append(A)            
          print('FedAvg evaluation')
          B = evaluate(server_state_FedAvg, central_test)
          history_FedAvg.append(B)  
          print('FedProx evaluation')
          C = evaluate(server_state_FedProx, central_test)
          history_FedProx_server_state.append(C)
          print('EF21 evaluation')
          D = evaluate(server_state_EF21, central_test)
          history_EF21_server_state.append(D)
          print('EC evaluation')
          _E_ = evaluate(server_state_EC, central_test)
          history_EC_server_state.append(E)

          log_writer(experiment_name,
                      history_FLARE,
                      history_FedAvg,
                      history_FedProx_server_state,
                      history_EF21_server_state,
                      history_EC_server_state,
                      prun_precent_logger_FLARE,
                      E_logger_FLARE,
                      prun_precent_logger_FFL,
                      E_logger_FFL,
                      output_list, A,B,C,D,_E_,round)
      
      # #apply FFL optiononal for Error Correcton
      # R_FFL, E_FFL =  calc_multypliers_FFL(history_FLARE,
      #                                     second_algo_server_state,
      #                                     R,
      #                                     E,
      #                                     central_test)                       
      
      tau_decay = tau_decay*c
      learning_rate  = [config.lr for i in range(NUM_CLIENTS)]

      #FLARE
      tau_cleints = [TAU/tau_decay for i in range(NUM_CLIENTS)]  
      t_cleints = [t for i in range (NUM_CLIENTS)] 
      clients_R_FLARE = [R for i in range(NUM_CLIENTS)]                   
      cleints_E_FLARE = [E for i in range(NUM_CLIENTS)]                            
      #FedProx
      clients_R_FedProx = [R for i in range(NUM_CLIENTS)]
      cleints_E_FedProx = [E for i in range(NUM_CLIENTS)]
      #EF21
      clients_R_EF21 = [R for i in range(NUM_CLIENTS)]
      cleints_E_EF21 = [E for i in range(NUM_CLIENTS)]  
      #EC
      clients_R_EC = [R for i in range(NUM_CLIENTS)]
      cleints_E_EC = [E for i in range(NUM_CLIENTS)]   
      #FedAvg
      cleints_E = [E for i in range(NUM_CLIENTS)]

      #loggers
      prun_precent_logger_FLARE.append(clients_R_FLARE[0])    
      E_logger_FLARE.append(cleints_E_FLARE[0])                            
      prun_precent_logger_FFL.append(clients_R_FedProx[0])
      E_logger_FFL.append(cleints_E_FedProx[0])      
      prun_precent_logger_EF21.append(clients_R_EF21[0])
      E_logger_EF21.append(cleints_E_EF21[0])  
      prun_precent_logger_EC.append(clients_R_EC[0])
      E_logger_EC.append(cleints_E_EC[0]) 

      # #Choose cleints randomly option
      # shufled_clients = train_dataset.client_ids
      # random.shuffle(shufled_clients)
      # client_ids = shufled_clients[:NUM_CLIENTS]
      # federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x)) for x in client_ids]      

      #Flare
      server_state_FLARE, accumolators_FLARE = next_fn(server_state_FLARE, accumolators_FLARE, federated_train_data, clients_R_FLARE, learning_rate, cleints_E_FLARE,tau_cleints,t_cleints)      
      #FedAvg
      server_state_FedAvg = FedAvg_next_fn(server_state_FedAvg, federated_train_data, cleints_E)       
      #Error Correcton algo 2 - FedProx
      server_state_FedProx, accumolators_FedProx = Second_algo_next_fn(server_state_FedProx, accumolators_FedProx, federated_train_data, clients_R_FedProx, learning_rate, cleints_E_FedProx,tau_cleints,t_cleints)
      #Error Correcton algo 3 - EF21
      server_state_EF21, accumolators_EF21 = Third_algo_next_fn(server_state_EF21, accumolators_EF21, federated_train_data, clients_R_EF21, learning_rate, cleints_E_EF21)
      #Error Correcton algo 4 - EC and TopK.
      server_state_EC, accumolators_EC = Fourth_algo_next_fn(server_state_EC, accumolators_EC, federated_train_data, clients_R_EC, learning_rate, cleints_E_EC)
      
      # #----------------- plot experiment -----------------------
      # if round % 10 == 0 :
      #     exp_plotter(history_FLARE,
      #                 history_FedAvg,
      #                 history_FedProx_server_state,
      #                 prun_precent_logger_FLARE,
      #                 prun_precent_logger_FFL,
      #                 E_logger_FLARE,
      #                 E_logger_FFL,
      #                 ROUNDS,
      #                 experiment_name)

# %%
# measure in numbers, how much non zeros weights in a layer fater sparsification during training

# pruned = tf.nest.map_structure(lambda x: sparsify_layer(x, 0.001),
#                             server_state_FLARE)
# #print(pruned.trainable[0])  #print first layer prunned
# #print number of non zero and nu,ber of total weights of all layers
# for i in range(len(pruned.trainable)):
  
#   print('number of non zeros weights: ',tf.math.count_nonzero(pruned.trainable[i]).numpy() , '     total weights number in layer:' , pruned.trainable[i].numpy().size  , '     layer shape: ',tf.shape(pruned.trainable[i]).numpy())

# pruned.trainable[1]
