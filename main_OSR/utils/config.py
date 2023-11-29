#global parms
TOTAL_NUM_CLIENTS = 100
NUM_CLIENTS = 10
BATCH_SIZE = 600  
lr = 0.001
MOMENTUM = 0.99

#calc_multypliers parms
begin_acc_realse = 5
end_acc_realse = 1000
update_acc_realse = 2
max_prec_to_prun = 0.001
max_E = 17
m_treshhold = 0.01 #1e-1
early_stop = 0.9

prun_percent_multyplier_const = 10 #this is the abount to divide the prun precent each time my algo decides so
E_multyplier_const = 1           #this is the abount to multply the E each time my algo decides so

#Reg parms
tau_decay_const = 1.05
reg_power = 2
inc_u_every = 50