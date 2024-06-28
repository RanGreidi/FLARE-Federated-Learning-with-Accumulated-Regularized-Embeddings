import os
def log_writer(experiment_name,
                history_federeted,
                history_FedAvg,
                history_second_algo_server_state,
                history_third_algo_server_state,
                history_fourth_algo_server_state,
                prun_precent_logger,
                E_logger,
                prun_precent_logger_FFL,
                E_logger_FFL,
                output_list, a,b,c,d,e,round):

    #write FLARE loss to text file
    with open(os.path.join('results/' + experiment_name, 'FLARE_evaluation'), "w") as FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_federeted)
        FedAlgo_file.write(toFile)
    #write FedAvg loss to text file
    with open(os.path.join('results/' + experiment_name, 'FedAvg_evaluation'), "w") as Fedavg_file:
        toFile = "\n".join(str(item) for item in history_FedAvg)
        Fedavg_file.write(toFile)  
    #write Second Algo loss to text file
    with open(os.path.join('results/' + experiment_name, 'FedProx evaluation'), "w") as Second_FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_second_algo_server_state)
        Second_FedAlgo_file.write(toFile)   
    #write Third Algo loss to text file
    with open(os.path.join('results/' + experiment_name, 'EF21_evaluation'), "w") as Third_FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_third_algo_server_state)
        Third_FedAlgo_file.write(toFile) 
    #write fourth Algo loss to text file
    with open(os.path.join('results/' + experiment_name, 'EC_and_TopK'), "w") as Third_FedAlgo_file:
        toFile = "\n".join(str(item) for item in history_fourth_algo_server_state)
        Third_FedAlgo_file.write(toFile) 
    #write prun_precent_logger to text file
    with open(os.path.join('results/' + experiment_name, 'FLARE_Sparse_precent_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in prun_precent_logger)
        outputfile.write(toFile)

    #write E_logger to text file
    with open(os.path.join('results/' + experiment_name, 'FLARE_E_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in E_logger)
        outputfile.write(toFile)

    #write FFL prun_precent_logger to text file
    with open(os.path.join('results/' + experiment_name, 'SecondAlgo_Sparse_precent_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in prun_precent_logger_FFL)
        outputfile.write(toFile)

    #write FFL E_logger to text file
    with open(os.path.join('results/' + experiment_name, 'SecondAlgo_E_logger'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in E_logger_FFL)
        outputfile.write(toFile)

    #write output list for linux terminal
    output_list.append(str(
                        "round: " + str(round) + '\n'
                        'FLARE evaluation:   ' + str(a[1]) + '\n'
                        'FedAvg evaluation:   '        + str(b[1]) + '\n'
                        'second_algo_server_state:   '     + str(c[1]) + '\n'
                        'third_algo_server_state:   '     + str(d[1]) + '\n' +'\n'
                        ))
    with open(os.path.join('results/' + experiment_name, 'output'), "w") as outputfile:
        toFile = "\n".join(str(item) for item in output_list)
        outputfile.write(toFile)
    #print('\n')