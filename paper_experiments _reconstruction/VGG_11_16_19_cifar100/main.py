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
              'ROUNDS':               [100000,1001,1001,1001,51,51,1001,1001,201],
              'R' :                   [0.001,0.001,0.001,0.001,0.001,0.0001,0.0001,0.0001,0.0001],
              'E':                    [1,1,1,1,1,5,5,16,1],
              'TAU':                  [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0],             
              'c':                    [1.01,1.01,1.001,1.001,1.001,1.01,1.01,1.01,0],
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

    experiment_name = '{}R_{}E_{}TAU_{}CLIENTS_{}ROUNDS_{}Decay_{}t_{}p_lr{}_vgg19_cifar100'.format(R,E,TAU,NUM_CLIENTS,ROUNDS,c,t,config.p,config.lr)
    exp_name_list.append(experiment_name)
    
    #make dir for experiment results
    if not os.path.exists('results/' + experiment_name): 
        os.makedirs('results/' + experiment_name)

    #initialize expiriment
    history_federeted = []
    history_FedAvg = []
    history_second_algo_server_state = []
    history_third_algo_server_state = []
    history_fourth_algo_server_state = []

    output_list = []
    prun_precent_logger = []
    E_logger = []
    prun_precent_logger_FFL = []
    E_logger_FFL = []
    prun_precent_logger_third = []
    E_logger_third = []
    prun_precent_logger_fourth = []
    E_logger_fourth = []

    server_state = initial_server_state
    accumolators = acc_init()
    accumolators_second_algo = acc_init()
    accumolators_third_algo = acc_init()
    accumolators_fourth_algo = acc_init()

    server_state_FedAvg = server_state
    server_state_FLARE = server_state
    second_algo_server_state = server_state
    third_algo_server_state = server_state
    fourth_algo_server_state = server_state

    tau_decay = 1

    for round in range(ROUNDS):
      #evaluate every 10 rounds
      print("round:",round)
      if round % 50 == 0 :           
          print('FLARE evaluation')
          A = evaluate(server_state_FLARE, central_test)
          history_federeted.append(A)            
          print('FedAvg evaluation')
          B = evaluate(server_state_FedAvg, central_test)
          history_FedAvg.append(B)  
          print('FedProx evaluation')
          C = evaluate(second_algo_server_state, central_test)
          history_second_algo_server_state.append(C)
          print('EF21 evaluation')
          D = evaluate(third_algo_server_state, central_test)
          history_third_algo_server_state.append(D)
          print('Fourth ALgo evaluation')
          _E_ = evaluate(fourth_algo_server_state, central_test)
          history_fourth_algo_server_state.append(E)

          log_writer(experiment_name,
                      history_federeted,
                      history_FedAvg,
                      history_second_algo_server_state,
                      history_third_algo_server_state,
                      history_fourth_algo_server_state,
                      prun_precent_logger,
                      E_logger,
                      prun_precent_logger_FFL,
                      E_logger_FFL,
                      output_list, A,B,C,D,_E_,round)
      
      # #apply FFL optiononal for Error Correcton
      # R_FFL, E_FFL =  calc_multypliers_FFL(history_federeted,
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
      #Error Correcton algo 2 - FedProx
      clients_R_second_algo = [R for i in range(NUM_CLIENTS)]
      cleints_E_second_algo = [E for i in range(NUM_CLIENTS)]
      #Error Correcton + sparsification
      clients_R_third_algo = [R for i in range(NUM_CLIENTS)]
      cleints_E_third_algo = [E for i in range(NUM_CLIENTS)]  
      #Error Correcton algo 4
      clients_R_fourth_algo = [R for i in range(NUM_CLIENTS)]
      cleints_E_fourth_algo = [E for i in range(NUM_CLIENTS)]   
      #FedAvg
      cleints_E = [E for i in range(NUM_CLIENTS)]

      #loggers
      prun_precent_logger.append(clients_R_FLARE[0])    
      E_logger.append(cleints_E_FLARE[0])                            
      prun_precent_logger_FFL.append(clients_R_second_algo[0])
      E_logger_FFL.append(cleints_E_second_algo[0])      
      prun_precent_logger_third.append(clients_R_third_algo[0])
      E_logger_third.append(cleints_E_third_algo[0])  
      prun_precent_logger_fourth.append(clients_R_fourth_algo[0])
      E_logger_fourth.append(cleints_E_fourth_algo[0]) 

      # #Choose cleints randomly option
      # shufled_clients = train_dataset.client_ids
      # random.shuffle(shufled_clients)
      # client_ids = shufled_clients[:NUM_CLIENTS]
      # federated_train_data = [preprocess(train_dataset.create_tf_dataset_for_client(x)) for x in client_ids]      

      #Flare
      server_state_FLARE, accumolators = next_fn(server_state_FLARE, accumolators, federated_train_data, clients_R_FLARE, learning_rate, cleints_E_FLARE,tau_cleints,t_cleints)      
      #FedAvg
      server_state_FedAvg = FedAvg_next_fn(server_state_FedAvg, federated_train_data, cleints_E)       
      #Error Correcton algo 2 - FedProx
      second_algo_server_state, accumolators_second_algo = Second_algo_next_fn(second_algo_server_state, accumolators_second_algo, federated_train_data, clients_R_second_algo, learning_rate, cleints_E_second_algo,tau_cleints,t_cleints)
      #Error Correcton algo 3 - EF21
      third_algo_server_state, accumolators_third_algo = Third_algo_next_fn(third_algo_server_state, accumolators_third_algo, federated_train_data, clients_R_third_algo, learning_rate, cleints_E_third_algo)
      #Error Correcton algo 4 - Currently implementing EC and TopK.
      fourth_algo_server_state, accumolators_fourth_algo = Fourth_algo_next_fn(fourth_algo_server_state, accumolators_fourth_algo, federated_train_data, clients_R_fourth_algo, learning_rate, cleints_E_fourth_algo)

      
      # #----------------- plot experiment -----------------------
      # if round % 10 == 0 :
      #     exp_plotter(history_federeted,
      #                 history_FedAvg,
      #                 history_second_algo_server_state,
      #                 prun_precent_logger,
      #                 prun_precent_logger_FFL,
      #                 E_logger,
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
