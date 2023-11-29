import os
def log_writer(experiment_name,
                history_federeted,
                history_FedAvg,
                history_second_algo_server_state,
                prun_precent_logger,
                E_logger,
                prun_precent_logger_FFL,
                E_logger_FFL,
                output_list, a,b,c,round):

    #write MyAlgo loss to text file
    with open(os.path.join('results/' + experiment_name, 'FederatedAlgo_evaluation'), "w") as FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_federeted)
        FedAlgo_file.write(toFile)
    #write FedAvg loss to text file
    with open(os.path.join('results/' + experiment_name, 'FedAvg_evaluation'), "w") as Fedavg_file:
        toFile = "\n".join(str(item) for item in history_FedAvg)
        Fedavg_file.write(toFile)  
    #write Second Algo loss to text file
    with open(os.path.join('results/' + experiment_name, 'Second_algo__evaluation'), "w") as Second_FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_second_algo_server_state)
        Second_FedAlgo_file.write(toFile)   

    #write prun_precent_logger to text file
    with open(os.path.join('results/' + experiment_name, 'prun_precent_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in prun_precent_logger)
        outputfile.write(toFile)

    #write E_logger to text file
    with open(os.path.join('results/' + experiment_name, 'E_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in E_logger)
        outputfile.write(toFile)

    #write FFL prun_precent_logger to text file
    with open(os.path.join('results/' + experiment_name, 'prun_precent_logger_FFL'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in prun_precent_logger_FFL)
        outputfile.write(toFile)

    #write FFL E_logger to text file
    with open(os.path.join('results/' + experiment_name, 'E_logger_FFL'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in E_logger_FFL)
        outputfile.write(toFile)

    #write output list for linux terminal
    output_list.append(str(
                        "round: " + str(round) + '\n'
                        'FederatedAlgo evaluation:   ' + str(a[1]) + '\n'
                        'FedAvg evaluation:   '        + str(b[1]) + '\n'+'\n'
                        'second_algo_server_state:   '     + str(c[1]) + '\n'+'\n'
                        ))
    with open(os.path.join('results/' + experiment_name, 'output'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in output_list)
        outputfile.write(toFile)
    #print('\n')