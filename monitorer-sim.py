from subprocess import run, PIPE, Popen, TimeoutExpired
import argparse
import os
import numpy as np
import networkx as nx
import pickle 

many_graph_instances = True

epochs, lr, batch_size = 500, 1e-4, 1
train_val_test_ratio = [6e-1, 2e-1, 2e-1]

n_I = [2]
trials_per_number = 200

beta, gamma = 0.2, 0.1
deltaT, maxTime, sim = 0.5, 20, int(10e3)

#hidden_dim_array = [8]
hidden_dim_array = [64]

datasets_array = ['./real_graphs/karate']
model = 'ode_nn'
out_of_dist = False

if not many_graph_instances:
    scriptName = "./ode_nn.py"
elif many_graph_instances and model!='ode_nn' and model!='dmp':
    scriptName = "./gnn_ngraph.py"
elif model=='dmp':
    scriptName = "./dmp.py"
else:
    scriptName = "./ode_nn_ngraph_sim.py"

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
    out_of_dist=None,
): 
    base = ["python3", scriptName]

    if lr != None:
        base += ["--lr", str(lr)]

    if epochs != None:
        base += ["--epochs", str(epochs)]

    if hidden != None:
        base += ["--hidden", str(hidden)]

    if I_indices != None:
        base += ["--I_indices"] + [str(i) for i in I_indices]

    if beta != None:
        base += ["--beta"] + [str(i) for i in beta]
    
    if gamma != None:
        base += ["--gamma"]+ [str(i) for i in gamma]

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

    if out_of_dist != None and out_of_dist != False:
        base += ["--out_of_dist"]

    return base

def random_parameters_SIR(graph_label, n_I, trials_per_number):
    G = pickle.load(open(graph_label + ".pkl", "rb"))
    G = G.to_undirected()
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)

    nodes = len(G.nodes())

    I_indices_array, beta_array, gamma_array = [], [], []
    for i in n_I:
        for _ in range(trials_per_number):
            indices = np.random.choice(nodes, i, replace=False)
            I_indices_array.append(list(indices))
            beta_array.append(np.random.uniform(0.1,0.5))
            gamma_array.append(np.random.uniform(0.1,0.5))

    return I_indices_array, beta_array, gamma_array

def main():
    parser = argparse.ArgumentParser()
    # To make the input integers
    parser.add_argument('--only', nargs='+', type=int, default=[])
    opt = parser.parse_args()

    if not many_graph_instances:
        totalProcs = len(n_I) * len(hidden_dim_array) * trials_per_number
    else:
        totalProcs = len(hidden_dim_array)

    procedures = []
    procNum, trial = 1, 1
    print(totalProcs)

    I_indices_array, beta_array, gamma_array = [], [], []
    #I_indices_array = [[2], [59], [7], [22], [49, 25], [14, 47], [23, 15], [40, 3], [53, 29], [5, 31, 58, 19, 33], [52, 12, 46, 36, 23], [49, 26, 14, 1, 35], [60, 10, 11, 13, 48], [34, 37, 27, 47, 21]]
    #beta_array = [0.4280457238832357, 0.5848423041373412, 0.5929367578370537, 0.4565879387205645, 0.5039891791381186, 0.42409521405473993, 0.42167420760130403, 0.4938462185018531, 0.45087407670108004, 0.4006303358652939, 0.5426920006508729, 0.5876176458465692, 0.5495297417398078, 0.41507612869364824]
    #gamma_array = [0.31489618481103665, 0.3672998703583091, 0.48856655684000294, 0.2789063642668024, 0.39095305219595566, 0.28651813006390303, 0.30151715303172555, 0.2649697507382967, 0.4113132939347085, 0.2153894720421624, 0.36622020304830016, 0.32909553512922735, 0.43986470339491307, 0.4248845435871645]

    for dataset in datasets_array:
         
        path_to_save='./multi-graph-1/Experiments-seed'+str(n_I[0])+'-'+dataset[14:]
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        else:
            I_indices_array = pickle.load(open(path_to_save + '/initial-seed.pkl', "rb"))
            beta_array = pickle.load(open(path_to_save + '/initial-beta.pkl', "rb"))
            gamma_array = pickle.load(open(path_to_save + '/initial-gamma.pkl', "rb"))
            print(len(beta_array))

        if not I_indices_array and not beta_array and not gamma_array:
            I_indices_array, beta_array, gamma_array = random_parameters_SIR(dataset, n_I, trials_per_number)
            print(I_indices_array)
            print(beta_array)
            print(gamma_array)

        if not many_graph_instances:
            for i, I_indices in enumerate(I_indices_array): 
                beta = [beta_array[i]]
                gamma = [gamma_array[i]]
                
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
                        I_indices=I_indices,
                        beta=beta,
                        gamma=gamma,
                        deltaT=deltaT,
                        maxTime=maxTime,
                        sim=sim,
                        path_to_save=path_to_save,
                        trial=trial,
                        dataset=dataset,
                        model=model,
                        out_of_dist=out_of_dist
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
        else:
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
                    I_indices=I_indices_array,
                    beta=beta_array,
                    gamma=gamma_array,
                    deltaT=deltaT,
                    maxTime=maxTime,
                    sim=sim,
                    path_to_save=path_to_save,
                    trial=trial,
                    dataset=dataset,
                    model=model,
                    out_of_dist=out_of_dist
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