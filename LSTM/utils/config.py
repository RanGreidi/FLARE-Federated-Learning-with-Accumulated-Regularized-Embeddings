import numpy as np
#global parms
TOTAL_NUM_CLIENTS = 100
NUM_CLIENTS = 10
BATCH_SIZE = 1  
lr = 0.001
MOMENTUM = 0.99
#Reg parms
p = 1
#data related
Input_shape = (784,)

vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
SEQ_LENGTH = 100
BUFFER_SIZE = 100  # For dataset shuffling