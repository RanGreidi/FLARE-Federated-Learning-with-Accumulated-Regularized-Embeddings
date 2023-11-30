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

    fig,ax = plt.subplots(1,1,figsize=(15,8),gridspec_kw={'height_ratios': [5]})
    x_axis = np.array([10*i for i in range(0,len(plot_fedAvG_acc))])      #if evalute every 10, change 10 here if evaluate_every is changed

    #fig 1s
    ax.plot(x_axis,plot_fedAvG_acc, label='FEDavg' ,color="orange",linestyle='dashed')  
    ax.plot(x_axis,plot_fed_alg_acc, label='My' ,color="red")
    ax.plot(x_axis,plot_second_algo_acc, label='FFL' ,color="blue")
    #ax.get_xaxis().set_ticks([])  # supress x axis label
    ax.legend(loc='lower left',fontsize=28)
    ax.set_ylabel("Top-1 Accuracy", fontsize=28)
    ax.set_xlabel("Round", fontsize=28)

    #supress lines plt frame of plot
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)

    ax.legend()


    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    #gray bars
    for i in range(int(ROUNDS/5),ROUNDS,int(ROUNDS/5)):
        ax.axvline(x=i, color='black', ls='dashed',linewidth=0.2)


    fig.tight_layout()
    plt.savefig('results/' + experiment_name + '.png', bbox_inches='tight')
    #plt.show()
    return None