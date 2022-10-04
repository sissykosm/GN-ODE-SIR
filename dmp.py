import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint as odeintscp
from sklearn.metrics import mean_absolute_error
import pickle 

import os
from os import path 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint as odeint

import ndlib.models.ModelConfig as mc
import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm

#from models import GCN, GIN
from ode_nn import sir_nx, sir_pandas, sir_torch, sir, runge_kutta_order4, get_sir_t_nodes, get_sir_t_nodes_torch, csv_trials, save_trial_to_csv, create_graph, get_train_test, get_labels_from_idx

#torch.manual_seed(0)
#np.random.seed(0)

from functools import reduce
import networkx as nx
import torch as T
from torch_scatter import scatter
from torch_geometric.utils import degree

import scipy.sparse as sp

def cave_index(src_nodes, tar_nodes):
    edge_list = [(int(s), int(t)) for s, t in zip(src_nodes, tar_nodes)] #+ [(int(t), int(s)) for s, t in zip(src_nodes, tar_nodes)]
    E = len(edge_list)
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    attr = {edge:w for edge, w in zip(edge_list, range(E))}
    nx.set_edge_attributes(G, attr, "idx")

    cave = []
    for edge in edge_list:
        if G.has_edge(*edge[::-1]):
            cave.append(G.edges[edge[::-1]]["idx"])
        else:
            cave.append(E)
    return cave

def edgeList_ic(weight_adj):
    """From weighted adj generate a [4, 2*E] edge list. edge_list[2] is the directed edge weight  

    Args:
        weight_adj (np.array): weighted adj

    Returns:
        [type]: [description]
    """
    sp_mat = sp.coo_matrix(weight_adj)
    weight = sp_mat.data
    cave = cave_index(sp_mat.row, sp_mat.col)
    edge_list = np.vstack((sp_mat.row, sp_mat.col, weight, cave))
    return edge_list

def edgeList(weight_adj):
    sp_mat = sp.coo_matrix(weight_adj)
    weight = sp_mat.data
    cave = cave_index(sp_mat.row, sp_mat.col)
    edge_list = np.vstack((sp_mat.row, sp_mat.col, weight, cave))
    return edge_list

class DMP_SIR():
    def __init__(self, weight_adj, nodes_gamma): 
        self.edge_list = edgeList(weight_adj)
        # edge_list with size [3, E], (src_node, tar_node, weight) 
        self.src_nodes = T.LongTensor(self.edge_list[0])
        self.tar_nodes = T.LongTensor(self.edge_list[1])
        self.weights   = T.FloatTensor(self.edge_list[2])
        self.cave_index = T.LongTensor(self.edge_list[3])
        self.gamma = T.FloatTensor(nodes_gamma)[self.src_nodes]
        self.nodes_gamma = T.FloatTensor(nodes_gamma)
        
        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)
        self.marginals = []


    def mulmul(self, Theta_t):
        Theta = scatter(Theta_t, index=self.tar_nodes, reduce="mul", dim_size=self.N) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter(Theta_t, index=self.cave_index, reduce="mul", dim_size=self.E+1)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def _set_seeds(self, seed_list):
        self.seeds = T.zeros(self.N)
        self.seeds[seed_list] = 1

        # initial
        self.Ps_0 = 1 - self.seeds
        self.Pi_0 = self.seeds
        self.Pr_0 = T.zeros_like(self.seeds)

        self.Ps_i_0 = self.Ps_0[self.src_nodes]
        self.Pi_i_0 = self.Pi_0[self.src_nodes]
        self.Pr_i_0 = self.Pr_0[self.src_nodes]
        
        self.Phi_ij_0 = 1 - self.Ps_i_0
        self.Theta_ij_0 = T.ones(self.E)      

        # first iteration, t=1
        self.Theta_ij_t = self.Theta_ij_0 - self.weights * self.Phi_ij_0 + 1E-10 # get rid of NaN
        self.Ps_ij_t_1 = self.Ps_i_0 # t-1
        self.Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t) # t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_0 - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)
        self.Pr_t = self.Pr_0 + self.nodes_gamma*self.Pi_0
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_0, self.Pi_0, self.Pr_0])
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])

        # print(T.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))

    

    def iteration(self):
        self.Theta_ij_t = self.Theta_ij_t - self.weights * self.Phi_ij_t
        new_Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t)
        self.Ps_ij_t_1 = self.Ps_ij_t
        self.Ps_ij_t = new_Ps_ij_t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_t - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)
        self.Pr_t = self.Pr_t + self.nodes_gamma*self.Pi_t
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])

        # print(T.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))


    def _stop(self):
        I_former, R_former = self.marginals[-2][1:]
        I_later , R_later  = self.marginals[-1][1:]

        I_delta = T.sum(T.abs(I_former-I_later))
        R_delta = T.sum(T.abs(R_former-R_later))
        if I_delta>0.01 or R_delta>0.01:
            return False
        else:
            return True

    def output(self):
        marginals = [T.stack(m, dim=1) for m in self.marginals]
        marginals = T.stack(marginals, dim=0) 
        return marginals

    def run(self, seed_list, maxTime):
        self._set_seeds(seed_list)
        for time in range(maxTime-2):
            self.iteration()
            #if self._stop():
            #    break
        # Output a size of [T, N, 3] Tensor， T starts from t=1
        return self.output()

def load_SIR_labels(dataset, path_to_save, G, I_indices, beta, gamma, sim, maxTime):
    if os.path.exists(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + '.pkl'):
        S_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        I_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        R_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        print('ok')
    else:
        #S_per_sim, I_per_sim, R_per_sim, _, _ = sir_nx(G, I_indices, beta, gamma, sim, maxTime)
        S_per_sim, I_per_sim, R_per_sim = sir_torch(G, I_indices, beta, gamma, sim, maxTime)
        S_labels, I_labels, R_labels = S_per_sim[0]/sim, I_per_sim[0]/sim, R_per_sim[0]/sim
        
        #S_labels, I_labels, R_labels = np.mean(S_per_sim,0), np.mean(I_per_sim,0), np.mean(R_per_sim,0)
        pickle.dump(S_labels, open(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))
        pickle.dump(I_labels, open(path_to_save + '/' + dataset[14:] + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))
        pickle.dump(R_labels, open(path_to_save + '/' + dataset[14:] + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))

    return S_labels, I_labels, R_labels


def runge_kutta_baseline(A, args, test_loader):
    # baseline simple rk
    s_rk_time = time.time()
    loss = []

    I_indices_test = args.I_indices[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
    beta_test, gamma_test = args.beta[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):], args.gamma[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
    for i, data in enumerate(test_loader): 
        I_sampled_t, S_sampled_t, R_sampled_t = runge_kutta_order4(sir, A, np.shape(A)[0], I_indices_test[i], beta_test[i], gamma_test[i], args.deltaT, args.maxTime)

        loss_S = mean_absolute_error(S_sampled_t, torch.transpose(torch.squeeze(data[1], 0), 0, 1).cpu().numpy()[:,:,0])
        loss_I = mean_absolute_error(I_sampled_t, torch.transpose(torch.squeeze(data[1], 0), 0, 1).cpu().numpy()[:,:,1])
        loss_R = mean_absolute_error(R_sampled_t, torch.transpose(torch.squeeze(data[1], 0), 0, 1).cpu().numpy()[:,:,2])
        loss.append((loss_S + loss_I + loss_R)/3)

    print(loss)
    e_rk_time = time.time()
    print('Runge-kutta baseline Loss: {:.5f}'.format(np.mean(loss)))
    print('Time inference baseline: {:.5f}'.format(e_rk_time - s_rk_time))
    return np.mean(loss), (e_rk_time - s_rk_time)

import argparse
import time 

torch.set_default_dtype(torch.float64)
def main():

    # Argument parser
    parser = argparse.ArgumentParser(description='Neural ODE')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
    parser.add_argument('--sim', type=int, default=1000, metavar='N', help='Simulations for Monte-Carlo')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.2], metavar='N', help='Beta for SIR')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.1], metavar='N', help='Gamma for SIR')
    parser.add_argument('--deltaT', type=float, default=0.5, metavar='N', help='deltaT for SIR')
    parser.add_argument('--maxTime', type=int, default=20, metavar='N', help='maxTime for SIR')
    parser.add_argument("--I_indices", help="I_indices for SIR", nargs='+', default=[12])
    parser.add_argument('--hidden', type=int, default=32, metavar='N', help='Size of hidden layer of NN')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size')
    parser.add_argument("--path_to_save", help="Path to save plots", default="./plots")
    parser.add_argument('--trial', type=int, default=32, metavar='N', help='Number of trial')
    parser.add_argument("--dataset", help="Path to the dataset", default="none")
    parser.add_argument("--train_val_test_ratio", help="Ratio for train, val and test split from the initial time series", nargs=3, type=float, default=[5e-1, 1e-1, 4e-1])
    parser.add_argument('--model', default='ode_nn', help="model to train, choose between lstm, tcn, encdecgru", type=str)
    parser.add_argument('--out_of_dist', default=False, action='store_true')
    args = parser.parse_args()

    #G, A = create_graph(50, './real_graphs/fb-social')
    G, A, _ = create_graph(50, args.dataset)
    n_nodes = np.shape(A)[0]
    print(n_nodes)

    #L = nx.line_graph(G)
    #A_line = nx.to_numpy_matrix(L)

    args.I_indices = [i[1 : -1].split(', ') for i in args.I_indices]
    args.I_indices = [list(map(int, i)) for i in args.I_indices]

    if not os.path.exists(args.path_to_save + '/initial-seed.pkl'):
        pickle.dump(args.I_indices, open(args.path_to_save + '/initial-seed.pkl', "wb" ))
        pickle.dump(args.beta, open(args.path_to_save + '/initial-beta.pkl', "wb" ))
        pickle.dump(args.gamma, open(args.path_to_save + '/initial-gamma.pkl', "wb" ))

    S_labels, I_labels, R_labels = [], [], []
    S0, I0, R0 = [], [], []
    for i, I_indices in enumerate(args.I_indices): 
        beta = args.beta[i]
        gamma = args.gamma[i]

        # create labels for all nodes per t using monte-carlo simulations 
        S_mc, I_mc, R_mc = load_SIR_labels(args.dataset, args.path_to_save, G, I_indices, beta, gamma, args.sim, args.maxTime)
        S_labels.append(S_mc)
        I_labels.append(I_mc)
        R_labels.append(R_mc)

        # initial condition per seed 
        S0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes), dtype=torch.float)
        I0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes), dtype=torch.float)
        R0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes), dtype=torch.float)
        I0_per_I[0,list(I_indices)] = 1
        S0_per_I[0,:] = torch.ones((n_nodes,), dtype=torch.float) - I0_per_I[0,:]
        S0.append(S0_per_I)
        I0.append(I0_per_I)
        R0.append(R0_per_I)

    # create loaders for train, val, test (train includes 0.8 of the graph instances, val, test includes 0.2 each)

    labels = torch.transpose(torch.cat((torch.tensor(S_labels).unsqueeze(-1), torch.tensor(I_labels).unsqueeze(-1), torch.tensor(R_labels).unsqueeze(-1)), axis=-1), 1, 2)
    tensor_train_x, tensor_val_x, tensor_test_x = [], [], []
    tensor_train_y, tensor_val_y, tensor_test_y = [], [], []

    if args.out_of_dist==False:
        for i, I_indices in enumerate(args.I_indices): 
            beta_gamma_batch = torch.zeros(n_nodes, args.hidden, dtype=torch.float)
            beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
            if (i>=0) and (i <= int(args.train_val_test_ratio[0]*len(S_labels))-1):
                tensor_train_x.append(torch.cat((torch.unsqueeze(S0[i][0],1), torch.unsqueeze(I0[i][0],1), torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_train_y.append(labels[i])
            elif (i > int(args.train_val_test_ratio[0]*len(S_labels))-1) and (i <= int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(S_labels))-1):
                tensor_val_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_val_y.append(labels[i])
            else:
                tensor_test_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_test_y.append(labels[i])
    
    else:
        dict_idx = pickle.load(open(args.path_to_save + '/out-of-dist-gamma.pkl', "rb"))
        #count_val = 0
        idx_test = dict_idx['test']
        I_indices_test, beta_test, gamma_test = [], [], []
        for i, I_indices in enumerate(args.I_indices): 
            beta_gamma_batch = torch.zeros(n_nodes, args.hidden, dtype=torch.float)
            beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
            if i in dict_idx['train']:
                tensor_train_x.append(torch.cat((torch.unsqueeze(S0[i][0],1), torch.unsqueeze(I0[i][0],1), torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_train_y.append(labels[i])
            elif i in dict_idx['val']:
                tensor_val_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_val_y.append(labels[i])
            else:
                tensor_test_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1), beta_gamma_batch),-1))
                tensor_test_y.append(labels[i])

                I_indices_test.append(args.I_indices[i])
                beta_test.append(args.beta[i])
                gamma_test.append(args.gamma[i])    
   
    tensor_train_x = torch.stack(tensor_train_x)
    tensor_train_y = torch.stack(tensor_train_y)
    
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tensor_val_x = torch.stack(tensor_val_x)
    tensor_val_y = torch.stack(tensor_val_y)
    val_dataset = TensorDataset(tensor_val_x, tensor_val_y)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    tensor_test_x = torch.stack(tensor_test_x)
    tensor_test_y = torch.stack(tensor_test_y)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # build model
    criterion = nn.L1Loss()
    s_rk_time = time.time()
    loss_all = 0
    items = 0

    dgraph = nx.DiGraph(G)
    node_rename = {n:i for i, n in enumerate(dgraph.nodes())}
    dgraph = nx.relabel_nodes(dgraph, node_rename)

    if args.out_of_dist==False:
        I_indices_test = args.I_indices[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
        beta_test, gamma_test = args.beta[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):], args.gamma[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
    for i, data in enumerate(test_loader): 
        print('dmp')
        model = DMP_SIR(A*beta_test[i], [gamma_test[i]]*np.shape(A)[0])
        SIR = model.run(I_indices_test[i], args.maxTime)
        
        #print(data[1])
        #print(data[1].size())
        #print(data[1].view(-1, data[1].size(1), data[1].size(3)))
        #loss = criterion(SIR[1:,:,:], torch.transpose(data[1][0], 0, 1)[1:,:,:])
        loss = criterion(SIR[1:], torch.transpose(data[1][0], 0, 1)[1:])

        loss_all += loss*3*data[1].size()[1]*(data[1].size()[2]-1)
        items += 3*data[1].size()[1]*(data[1].size()[2]-1)
        print(loss)

    test_loss = loss_all/items
    
    e_rk_time = time.time()
    print('DMP baseline Loss: {:.5f}'.format(test_loss))
    print('Time inference baseline: {:.5f}'.format(e_rk_time - s_rk_time))
    
    if not path.exists(args.path_to_save + '/Metrics-trials2-'+ os.path.relpath(args.dataset, './real_graphs/')):
        loss_baseline, rk_time = 0, 0
        loss_baseline, rk_time = 0, 0

    # Keep results for all trials in csv
    #save_trial_to_csv(args, 0, 0, test_loss, loss_baseline, e_test_time - s_test_time, rk_time)
    
    return 

if __name__ == '__main__':
    main()