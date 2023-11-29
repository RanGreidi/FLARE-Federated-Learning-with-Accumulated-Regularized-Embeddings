from matplotlib import pyplot as plt
import numpy as np
def exp_plotter(history_federeted,
                history_FedAvg,
                history_second_algo_server_state,
                prun_precent_logger,
                prun_precent_logger_FFL,
                E_logger,
                E_logger_FFL,
                ROUNDS,
                experiment_name):

    plot_fed_alg_acc = []
    plot_fed_alg_loss = []
    for elem in history_federeted:
        loss=elem[0]
        acc=elem[1]
        plot_fed_alg_acc.append(acc)
        plot_fed_alg_loss.append(loss)

    plot_fedAvG_acc = []
    plot_fedAVG_loss = []
    for elem in history_FedAvg:
        loss=elem[0]
        acc=elem[1]
        plot_fedAvG_acc.append(acc)
        plot_fedAVG_loss.append(loss)

    plot_second_algo_acc = []
    plot_second_algo_loss = []
    for elem in history_second_algo_server_state:
        loss=elem[0]
        acc=elem[1]
        plot_second_algo_acc.append(acc)
        plot_second_algo_loss.append(loss)

    fig,ax = plt.subplots(3,1,figsize=(8,4),gridspec_kw={'height_ratios': [3, 1,1]})
    x_axis = np.array([10*i for i in range(0,len(plot_fedAvG_acc))])      #if evalute every 10, change 10 here if evaluate_every is changed

    #fig 1s
    ax[0].plot(x_axis,plot_fedAvG_acc, label='FEDavg' ,color="orange",linestyle='dashed')  
    ax[0].plot(x_axis,plot_fed_alg_acc, label='My' ,color="red")
    ax[0].plot(x_axis,plot_second_algo_acc, label='FFL' ,color="blue")
    ax[0].get_xaxis().set_ticks([])  # supress x axis label
    ax[0].legend()
    ax[0].title.set_text(' ')

    #supress lines plt frame of plot
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(True)
    ax[0].spines['left'].set_visible(True)

    #fig 2
    x_axis_2 = np.array([i for i in range(0,len(prun_precent_logger))])
    ax[1].plot(x_axis_2,np.log10(prun_precent_logger), label='My sparsity' ,color="red") 
    ax[1].plot(x_axis_2,np.log10(prun_precent_logger_FFL), label='FFL sparsity' ,color="blue") 

    #supress lines plt frame of plot
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(True)
    ax[1].spines['left'].set_visible(True)
    ax[1].legend()

    #fig 3
    x_axis_2 = np.array([i for i in range(0,len(E_logger))])
    ax[2].plot(x_axis_2,E_logger, label='My E' ,color="red") 
    ax[2].plot(x_axis_2,E_logger_FFL, label='FFL E' ,color="blue") 

    #supress lines plt frame of plot
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_visible(True)
    ax[2].spines['left'].set_visible(True)

    ax[2].legend()

    #gray bars
    for i in range(int(ROUNDS/4),ROUNDS,int(ROUNDS/4)):
        ax[0].axvline(x=i, color='black', ls='dashed',linewidth=0.2)
        ax[1].axvline(x=i, color='black', ls='dashed',linewidth=0.2)
        ax[2].axvline(x=i, color='black', ls='dashed',linewidth=0.2)

    fig.tight_layout()
    #plt.savefig('results/' + experiment_name + '.png', bbox_inches='tight')
    plt.show()

    return None