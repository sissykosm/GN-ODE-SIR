from subprocess import run, PIPE, Popen, TimeoutExpired
import argparse
import os
import numpy as np
import networkx as nx
import pickle 

many_graph_instances = True

epochs, lr, batch_size = 500, 1e-3, 8
train_val_test_ratio = [6e-1, 2e-1, 2e-1]

n_I = [2]
trials_per_number = 200

beta, gamma = 0.2, 0.1
deltaT, maxTime, sim = 0.5, 20, int(10e3)

#hidden_dim_array = [8]
hidden_dim_array = [8, 8, 8, 8]

datasets_array = ['./real_graphs/dolphins+fb-food+fb-social+openflights+wiki-vote+epinions']
model = 'GIN'

if not many_graph_instances:
    scriptName = "./ode_nn.py"
elif many_graph_instances and model!='ode_nn':
    scriptName = "./gnn_ngraphs.py"
else:
    scriptName = "./ode_nn_ngraphs.py"

def createArgs(
    lr=None,
    epochs=None,
    hidden=None,
    batch_size=None,
    I_indices=None,
    beta=None,
    gamma=None,
    deltaT=None,
    maxTime=None,
    sim=None,
    dataset=None,
    trial=None,
    path_to_save=None,
    train_val_test_ratio=None,
    model=None,
): 
    base = ["python3", scriptName]

    if lr != None:
        base += ["--lr", str(lr)]

    if epochs != None:
        base += ["--epochs", str(epochs)]

    if hidden != None:
        base += ["--hidden", str(hidden)]

    if deltaT != None:
        base += ["--deltaT", str(deltaT)]
    
    if maxTime != None:
        base += ["--maxTime", str(maxTime)]
    
    if sim != None:
        base += ["--sim", str(sim)]

    if trial != None:
        base += ["--trial", str(trial)]

    if dataset != None:
        base += ["--dataset", dataset]

    if path_to_save != None:
        base += ["--path_to_save", path_to_save]
    
    if batch_size != None:
        base += ["--batch_size", str(batch_size)]

    if train_val_test_ratio != None:
        base += ["--train_val_test_ratio"] + [str(i) for i in train_val_test_ratio]

    if model != None:
        base += ["--model", model]

    return base

def main():
    parser = argparse.ArgumentParser()
    # To make the input integers
    parser.add_argument('--only', nargs='+', type=int, default=[])
    opt = parser.parse_args()

    if many_graph_instances:
        totalProcs = len(hidden_dim_array)

    procedures = []
    procNum, trial = 1, 1
    print(totalProcs)

    for dataset in datasets_array:
         
        path_to_save='./multi-graph-1/Experiments-seed'+str(n_I[0])+'-ngraphs4'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
        if many_graph_instances:
            for hidden in hidden_dim_array: 
                if len(opt.only) > 0:
                    if procNum not in opt.only:
                        procNum += 1
                        continue

                args = createArgs(
                    lr=lr,
                    epochs=epochs,
                    hidden=hidden,
                    train_val_test_ratio=train_val_test_ratio,
                    batch_size=batch_size,
                    deltaT=deltaT,
                    maxTime=maxTime,
                    sim=sim,
                    path_to_save=path_to_save,
                    trial=trial,
                    dataset=dataset,
                    model=model,
                )
                print(args)

                proc = Popen(args)
                procedures.append(proc)
                
                print("[MONITORER] Started neural network " + str(procNum) + "/" + str(totalProcs))
                
                returnCode = proc.wait()
                if returnCode != 0:
                    print("[MONITORER] Oops! Something broke!")
                
                procNum += 1
                trial +=1
        trial=1

    print("Spawned " + str(len(procedures)) + " processes.")

if __name__ == "__main__":
    main()