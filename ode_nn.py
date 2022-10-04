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
from torchdiffeq import odeint_adjoint as odeint

import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics import SIRModel
#import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm

from models import GCN, GIN

#torch.manual_seed(0)
#np.random.seed(0)

import time 
import pandas as pd

def sir_torch(G, seed_set, beta, gamma, sims=10000, T=20): 
    n = G.number_of_nodes()
    edges = np.zeros((2*G.number_of_edges(), 2), dtype=np.int64)
    for i,edge in enumerate(G.edges()):
        edges[2*i,0] = edge[0]
        edges[2*i,1] = edge[1]

        edges[(2*i)+1,0] = edge[1]
        edges[(2*i)+1,1] = edge[0]

    start = time.time()
    edges = torch.LongTensor(edges).cuda()
    I_per_sim = torch.zeros((1,T,n)).cuda()
    S_per_sim = torch.zeros((1,T,n)).cuda()
    R_per_sim = torch.zeros((1,T,n)).cuda()

    for i in range(sims):
        # Simulate propagation process      
        I = torch.zeros(n).cuda()
        S = torch.ones(n).cuda()
        R = torch.zeros(n).cuda()

        I[seed_set] = 1 
        S[seed_set] = 0

        I_per_sim[:,0,:] = I
        S_per_sim[:,0,:] = S

        for it in range(1, T):
            # For each newly active node, find its neighbors that become activated
            idx_I = torch.where(I==1)[0]
            targets = edges[torch.isin(edges[:,0], idx_I),:]
            targets = targets[torch.isin(targets, torch.where(S==1)[0])]      

            # Determine the neighbors that become infected
            coins  = torch.rand(targets.size(0))
            choice = torch.where(coins < beta)[0]
            new_infected = targets[choice]

            # Determine the Infected nodes that become Recovered
            coins  = torch.rand(idx_I.size(0))
            choice = torch.where(coins < gamma)[0]
            new_recovered = idx_I[choice]
            R[new_recovered] = 1

            # Add newly activated nodes to the set of activated nodes
            I[new_infected] = 1
            I[new_recovered] = 0
            S[new_infected] = 0
            
            I_per_sim[:,it,:] += I
            S_per_sim[:,it,:] += S
            R_per_sim[:,it,:] += R
            
    print("Time per graph ", time.time()-start)
    I_per_sim = I_per_sim.cpu().numpy()
    S_per_sim = S_per_sim.cpu().numpy()
    R_per_sim = R_per_sim.cpu().numpy()
    return S_per_sim, I_per_sim, R_per_sim

def sir_pandas(G, seed_set, beta, gamma, sims=10000, T=20): 
    n_nodes = G.number_of_nodes()
    source = []
    target = []
    for edge in G.edges():
        source.append(edge[0])
        target.append(edge[1])

        source.append(edge[1])
        target.append(edge[0])

        
    df = pd.DataFrame(list(zip(source, target)), columns =['source', 'target'])
    start = time.time()
    I_per_sim = np.zeros((sims,T,n_nodes))
    S_per_sim = np.zeros((sims,T,n_nodes))
    R_per_sim = np.zeros((sims,T,n_nodes))

    for i in range(sims):
        # Simulate propagation process      
        I = set(seed_set)
        S = set(range(n_nodes)) - I
        R = set()

        I_per_sim[i,0,list(I)] = 1
        S_per_sim[i,0,list(S)] = 1
        #R_per_sim[i,0,list(R)] = 1

        # if i==0:
        #     print("S ", S)
        #     print("I ", I)
        #     print("R ", R)
        for it in range(1, T):
            source = df.loc[df['source'].isin(I)]

            # For each newly active node, find its neighbors that become activated
            targets = np.array(source['target'].tolist())

            # Determine the neighbors that become infected
            coins  = np.random.random_sample((len(targets),))
            choice = np.where(coins < beta)[0]
            new_infected = set(targets[choice]) & S

            # Determine the Infected nodes that become Recovered
            coins  = np.random.random_sample((len(I),))
            choice = np.where(coins < gamma)[0]
            new_recovered = np.array(list(I))[choice]
            #choice = [gamma > coins[c] for c in range(len(coins))]
            #new_recovered = np.extract(choice, I)
            R = R.union(set(new_recovered))

            # Add newly activated nodes to the set of activated nodes
            I = I.union(new_infected)
            S = S - new_infected #- R
            I = I - set(new_recovered)
            
            I_per_sim[i,it,list(I)] = 1
            S_per_sim[i,it,list(S)] = 1
            R_per_sim[i,it,list(R)] = 1
            # if it<=5 and i==0:
            #     print('new I ', new_infected)
            #     print('new R', new_recovered)
            #     print("S ", S)
            #     print("I ", I)
            #     print("R ", R)

    print("Time per graph ", time.time()-start)  
    return S_per_sim, I_per_sim, R_per_sim

def sir_nx(G,seed_set, beta_factor, gamma_factor, sims=10000, num_steps=1000):
    """
    Given the graph and the seed set, compute the number of infected nodes after the end of a spreading process
    Input: networkx, set of seed nodes
    Output: average number of nodes influenced by the seed nodes
    """
    count = 0
    counts_per_sim = np.zeros((num_steps,sims))
    I_per_sim = np.zeros((sims,num_steps,np.shape(G)[0]))
    S_per_sim = np.zeros((sims,num_steps,np.shape(G)[0]))
    R_per_sim = np.zeros((sims,num_steps,np.shape(G)[0]))

    s_time = time.time()
    # run for sims simulations
    for i in range(sims):
        # Composite Model instantiation
        # Model selection
        model = SIRModel(G)

        # Model Configuration
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta_factor)
        cfg.add_model_parameter('gamma', gamma_factor)
        cfg.add_model_initial_configuration("Infected", seed_set)
        model.set_initial_status(cfg)
        iterations = model.iteration_bunch(num_steps)

        # calculate status per simulation and per t for S, I, R, 3-d matrices with 0,1 values
        indices_S, indices_I, indices_R = [], [], []
        for t in range(num_steps):
            indices_S = indices_S + [j for j, x in iterations[t]['status'].items() if x == 0]
            indices_I = indices_I + [j for j, x in iterations[t]['status'].items() if x == 1]
            indices_S = [i for i in indices_S if i not in indices_I]
            indices_R = indices_R + [j for j, x in iterations[t]['status'].items() if x == 2]
            indices_I = [i for i in indices_I if i not in indices_R]
            '''
            if t<=2 and i==0:
                print(iterations[t]['status'])
                print("S ", indices_S)
                print("I ", indices_I)
                print("R ", indices_R) '''
            S_per_sim[i,t,indices_S] = 1
            I_per_sim[i,t,indices_I] = 1
            R_per_sim[i,t,indices_R] = 1
        '''
        for j in range(0, num_steps):
            counts_per_sim[j,i] = iterations[j]["node_count"][1]
    
        count+=iterations[num_steps-1]["node_count"][1]'''
    e_time = time.time()
    print("Time per graph ", e_time-s_time)
    #return S_per_sim, I_per_sim, R_per_sim, np.mean(counts_per_sim, 1), count/sims
    return S_per_sim, I_per_sim, R_per_sim, 0, 0

# runge-kutta from scipy
def sir(x, y, A, beta, gamma):
    S, I, R = x[:np.shape(A)[0]], x[np.shape(A)[0]:2*np.shape(A)[0]], x[2*np.shape(A)[0]:]
    dS = np.squeeze(np.array(-beta*np.multiply(np.dot(A,I),S)))
    dI = np.squeeze(np.array(beta*(np.multiply(np.dot(A,I),S)) - gamma*I))
    dR = gamma*I
    return np.hstack(([dS, dI, dR]))

def runge_kutta_order4(sir, A, n_nodes, indices, beta_factor, gamma_factor, deltaT=1, maxTime = 70):
    beta = beta_factor
    gamma = gamma_factor * np.ones((n_nodes,))
    S, I, R = np.zeros((int(maxTime),n_nodes)), np.zeros((int(maxTime),n_nodes)), np.zeros((int(maxTime),n_nodes))
    I[0,indices] = 1
    S[0,:] = list(np.ones((n_nodes,))) - I[0,:]
    sol = odeintscp(sir, np.hstack(([S[0,:],I[0,:],R[0,:]])), np.arange(0,maxTime, deltaT), args=(A, beta, gamma))
    S_rk, I_rk, R_rk = sol[:,:np.shape(A)[0]], sol[:,np.shape(A)[0]:2*np.shape(A)[0]], sol[:,2*np.shape(A)[0]:]
    I_sampled_t = get_sir_t_nodes(I_rk, maxTime, deltaT, count=False)
    S_sampled_t = get_sir_t_nodes(S_rk, maxTime, deltaT, count=False)
    R_sampled_t = get_sir_t_nodes(R_rk, maxTime, deltaT, count=False)
    #print(I_sampled_t)
    return I_sampled_t, S_sampled_t, R_sampled_t

def get_sir_t_nodes(x_rk, maxTime, deltaT, count=True):

    if count==True:
        x_expected = np.sum(x_rk, axis=1)
        x_sampled_t = np.zeros((int(maxTime),))
        for i in range(maxTime):
            x_sampled_t[i] = x_expected[int(i/deltaT)]
    else:
        x_sampled_t = np.zeros((int(maxTime),np.shape(x_rk)[-1]))
        for i in range(maxTime):
            x_sampled_t[i,:] = x_rk[int(i/deltaT),:]

    return x_sampled_t

def get_sir_t_nodes_torch(x_rk, maxTime, deltaT, count=True):

    if count==True:
        x_expected = torch.sum(x_rk, axis=1)
        x_sampled_t = torch.zeros((int(maxTime),))
        for i in range(maxTime):
            x_sampled_t[i] = x_expected[int(i/deltaT)]
    else:
        x_sampled_t = torch.zeros((int(maxTime),x_rk.size()[-1]))
        for i in range(maxTime):
            x_sampled_t[i,:] = x_rk[int(i/deltaT),:]

    return x_sampled_t

class ODEfunc(nn.Module):
    def __init__(self, A, beta, gamma, hidden1, device):
        super(ODEfunc, self).__init__()
        self.A = torch.tensor(A, requires_grad=False).to(device)
        self.beta = beta
        self.gamma = gamma
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden1)
        self.bn3 = nn.BatchNorm1d(hidden1)
        self.ln = nn.LayerNorm(hidden1)
        self.linear = nn.Linear(hidden1, hidden1)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        x = self.linear(x)
        x = self.relu(x)

        S, I, R = x[:np.shape(self.A)[0],:], x[np.shape(self.A)[0]:2*np.shape(self.A)[0],:], x[2*np.shape(self.A)[0]:,:]
        dS = -self.beta*(torch.multiply(torch.mm(self.A,I),S))
        dI = self.beta*(torch.multiply(torch.mm(self.A,I),S)) - self.gamma*I
        dR = self.gamma*I

        # apply batch normalization 
        #dI = self.bn1(dI)
        #dS = self.bn2(dS)
        #dR = self.bn3(dR)

        # apply layer normalization 
        dS = self.ln(dS)
        dI = self.ln(dI)
        dR = self.ln(dR)
        return torch.cat((dS,dI,dR))

# define neural ode solver
class ODEBlock(nn.Module):

    def __init__(self, maxTime, deltaT, n_nodes, indices, hidden1, odefunc, device):
        super(ODEBlock, self).__init__()
        self.maxTime = maxTime
        self.deltaT = deltaT

        # SIR parameters
        self.maxTime = maxTime
        self.deltaT = deltaT
        self.integration_time = torch.from_numpy(np.arange(0, self.maxTime, self.deltaT)).to(device)
        self.odefunc = odefunc

        # graph parameters
        #self.A = torch.tensor(A, requires_grad=False)
        self.n_nodes = n_nodes
        self.indices = torch.tensor(indices, requires_grad=False)

        # model parameters

        # input linear layers
        self.hidden1 = hidden1
        self.linearI1 = nn.Linear(1, hidden1)
        self.linearS1 = nn.Linear(1, hidden1)

        # more linear layers
        self.linear3 = nn.Linear(hidden1, 4)
        self.relu3 = nn.ReLU()

        # output linear layers 
        self.linearI2 = nn.Linear(hidden1,1)
        self.linearS2 = nn.Linear(4,1)
        self.linearR2 = nn.Linear(hidden1,1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, S0, I0, R0): 
        
        S = self.linearS1(torch.unsqueeze(S0, -1)).double()
        I = self.linearS1(torch.unsqueeze(I0, -1)).double()
        S = self.relu(S)
        I = self.relu(I)
        
        sol = odeint(self.odefunc, torch.cat((S[0,:],I[0,:],R0[0,:])), self.integration_time, method='rk4')
        #print(sol)
        S, I, R = sol[:,:self.n_nodes], sol[:,self.n_nodes:2*self.n_nodes], sol[:,2*self.n_nodes:]

        S = self.linear3(S)
        I = self.linear3(I)
        R = self.linear3(R) 

        S = self.linearS2(self.relu3(S))
        I = self.linearS2(self.relu3(I))
        R = self.linearS2(self.relu3(R)) 
        
        SIR_vectors = torch.cat((S, I, R), -1)
        #print(SIR_vectors.size())

        #SIR_vectors = self.sigmoid(SIR_vectors)
        #SIR_vectors = SIR_vectors/torch.sum(SIR_vectors, dim=-1, keepdim=True)
        #S, I, R = SIR_vectors.chunk(3, dim=-1)
        
        SIR_vectors = self.softmax(SIR_vectors)
        S, I, R = SIR_vectors.chunk(3, dim=-1)
        return S, I, R
'''
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value '''

# create csv file with results for all trials from monitorer
# takes the list of column names and a list of values to write
def csv_trials(path_to_csv, columns, list_to_csv):
    if not path.exists(path_to_csv):
        with open(path_to_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerow(list_to_csv)
    else:
        with open(path_to_csv, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list_to_csv)
    import pandas as pd
    data = pd.read_csv(path_to_csv)
    print(data)
    return

def save_trial_to_csv(args, best_epoch, val_loss, test_loss, loss_baseline, n_ode_time, rk_time):
    list_to_csv = [args.trial, args.model, args.lr, args.epochs, args.sim, args.train_val_test_ratio, len(args.beta), len(args.gamma), args.deltaT, args.maxTime, [len(args.I_indices[0]), len(args.I_indices)], args.hidden, best_epoch, val_loss, test_loss, loss_baseline, n_ode_time, rk_time]
    csv_trials(args.path_to_save + '/Metrics-trials-'+ os.path.relpath(args.dataset, './real_graphs/'), ["trial", "model", "lr", "epochs", "MC sim", "train_val_test_ratio", "beta", "gamma", "deltaT", "maxTime", "I_indices", "hidden", "best_epoch", "val_loss", "test_loss", "loss_baseline", "n_ode_time", "rk_time"], list_to_csv)
    return 

def create_graph(n_nodes, graph_label='none'):

    if graph_label != 'none':
        G = pickle.load(open(graph_label + ".pkl", "rb"))
        G = G.to_undirected()
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
        print('nodes', len(G.nodes()))
        print('edges', len(G.edges()))
    else:
        #G = nx.cycle_graph(30)
        G = nx.fast_gnp_random_graph(n_nodes,0.2)

        #G = nx.complete_graph(n_nodes)
        '''
        for i in range(30):
            G.add_edge(i,i) '''

        #G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G)
    return G, A, 0#nx.to_numpy_matrix(G)

def get_train_test(n_nodes, train_test_ratio = [0.8, 0.1, 0.1]):
    #idx = np.random.permutation(n_nodes)
    idx = np.random.RandomState(seed=42).permutation(n_nodes)
    idx_train, idx_val, idx_test = idx[:int(train_test_ratio[0]*n_nodes)], idx[int(train_test_ratio[0]*n_nodes):int((train_test_ratio[0]+train_test_ratio[1])*n_nodes)], idx[int((train_test_ratio[0]+train_test_ratio[1])*n_nodes):]
    return idx_train, idx_val, idx_test

# X feature array of [t, n_nodes]
def get_labels_from_idx(X, idx):
    return X[:,idx]

def train(model, optimizer, criterion, device, idx_train, idx_val, S_train, I_train, R_train, S_val, I_val, R_val, maxTime, deltaT, S0, I0, R0, A, model_type):
    model.train()

    if model_type == 'ode_nn':
        S_pred, I_pred, R_pred = model(S0.to(device), I0.to(device), R0.to(device))

        S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
        I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
        R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)
    else:
        x_features = torch.cat((torch.unsqueeze(S0[0],1),torch.unsqueeze(I0[0],1),torch.unsqueeze(R0[0,:,0],1)),-1)
        S_pred_t, I_pred_t, R_pred_t = model(x_features.to(device), torch.LongTensor(np.array(np.nonzero(A))).to(device))
        S_pred_t, I_pred_t, R_pred_t = torch.squeeze(S_pred_t), torch.squeeze(I_pred_t), torch.squeeze(R_pred_t)

    #print(I_pred_t)
    loss_S = criterion(get_labels_from_idx(S_pred_t, idx_train).cpu(), torch.from_numpy(S_train))
    loss_I = criterion(get_labels_from_idx(I_pred_t, idx_train).cpu(), torch.from_numpy(I_train))
    loss_R = criterion(get_labels_from_idx(R_pred_t, idx_train).cpu(), torch.from_numpy(R_train))
    loss = (loss_S + loss_I + loss_R)/3
    #loss = loss_I
    #print(loss)
    #print(get_labels_from_idx(I_pred_t, idx_train[:4]))
    #print(torch.from_numpy(I_train)[:,:4])

    # test 

    val_loss_S = criterion(get_labels_from_idx(S_pred_t, idx_val).cpu(), torch.from_numpy(S_val))
    val_loss_I = criterion(get_labels_from_idx(I_pred_t, idx_val).cpu(), torch.from_numpy(I_val))
    val_loss_R = criterion(get_labels_from_idx(R_pred_t, idx_val).cpu(), torch.from_numpy(R_val))
    val_loss = (val_loss_S + val_loss_I + val_loss_R)/3
    #print(val_loss)
    #print(get_labels_from_idx(I_pred_t, idx_val))
    #print(torch.from_numpy(I_val))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, val_loss, get_labels_from_idx(S_pred_t, idx_train), get_labels_from_idx(I_pred_t, idx_train), get_labels_from_idx(R_pred_t, idx_train)

def test(model, criterion, device, idx_val, S_val, I_val, R_val, maxTime, deltaT, S0, I0, R0, A, model_type):
    model.eval()

    with torch.no_grad():
        if model_type == 'ode_nn':
            S_pred, I_pred, R_pred = model(S0.to(device), I0.to(device), R0.to(device))

            S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
            I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
            R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

        else:
            x_features = torch.cat((torch.unsqueeze(S0[0],1),torch.unsqueeze(I0[0],1),torch.unsqueeze(R0[0,:,0],1)),-1)
            S_pred_t, I_pred_t, R_pred_t = model(x_features.to(device), torch.LongTensor(np.array(np.nonzero(A))).to(device))
            S_pred_t, I_pred_t, R_pred_t = torch.squeeze(S_pred_t), torch.squeeze(I_pred_t), torch.squeeze(R_pred_t)

        loss_S = criterion(get_labels_from_idx(S_pred_t, idx_val).cpu(), torch.from_numpy(S_val))
        loss_I = criterion(get_labels_from_idx(I_pred_t, idx_val).cpu(), torch.from_numpy(I_val))
        loss_R = criterion(get_labels_from_idx(R_pred_t, idx_val).cpu(), torch.from_numpy(R_val))
        loss = (loss_S + loss_I + loss_R)/3
        #print(loss)
    
    return loss, get_labels_from_idx(S_pred_t, idx_val), get_labels_from_idx(I_pred_t, idx_val), get_labels_from_idx(R_pred_t, idx_val)

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
    args = parser.parse_args()

    #G, A = create_graph(50, './real_graphs/fb-social')
    G, _, A = create_graph(50, args.dataset)
    n_nodes = np.shape(A)[0]

    args.beta, args.gamma = args.beta[0], args.gamma[0]
    args.I_indices = list(map(int, args.I_indices))

    # create labels for all nodes per t using monte-carlo simulations 
    if os.path.exists(args.path_to_save + '/' + args.dataset[14:] + '-S-' + '-'.join(str(i) for i in args.I_indices) + '.pkl'):
        S_labels = pickle.load(open(args.path_to_save + '/' + args.dataset[14:] + '-S-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "rb"))
        I_labels = pickle.load(open(args.path_to_save + '/' + args.dataset[14:] + '-I-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "rb"))
        R_labels = pickle.load(open(args.path_to_save + '/' + args.dataset[14:] + '-R-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "rb"))
    else:
        S_per_sim, I_per_sim, R_per_sim, _, _ = sir_nx(G, args.I_indices, args.beta, args.gamma, args.sim, args.maxTime)
        S_labels, I_labels, R_labels = np.mean(S_per_sim,0), np.mean(I_per_sim,0), np.mean(R_per_sim,0)
        pickle.dump(S_labels, open(args.path_to_save + '/' + args.dataset[14:] + '-S-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "wb" ) )
        pickle.dump(I_labels, open(args.path_to_save + '/' + args.dataset[14:] + '-I-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "wb" ) )
        pickle.dump(R_labels, open(args.path_to_save + '/' + args.dataset[14:] + '-R-' + '-'.join(str(i) for i in args.I_indices) + ".pkl", "wb" ) )

    # get train, val, test indices randomly
    idx_train, idx_val, idx_test = get_train_test(n_nodes, args.train_val_test_ratio)

    # get labels for train, val, test
    S_train = get_labels_from_idx(S_labels, idx_train)
    S_val = get_labels_from_idx(S_labels, idx_val)
    S_test = get_labels_from_idx(S_labels, idx_test)

    I_train = get_labels_from_idx(I_labels, idx_train)
    I_val = get_labels_from_idx(I_labels, idx_val)
    I_test = get_labels_from_idx(I_labels, idx_test)

    R_train = get_labels_from_idx(R_labels, idx_train)
    R_val = get_labels_from_idx(R_labels, idx_val)
    R_test = get_labels_from_idx(R_labels, idx_test)
    
    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    if args.model == 'ode_nn':
        odefunc = ODEfunc(A, args.beta, args.gamma, args.hidden, device)
        model = ODEBlock(args.maxTime, args.deltaT, np.shape(A)[0], args.I_indices, args.hidden, odefunc, device)
    elif args.model == 'GCN':
        model = GCN(3, args.hidden, int(args.hidden/2), 3, 0.1, args.maxTime)
    else:
        model = GIN(3, args.hidden, int(args.hidden/2), 3, 0.1, args.maxTime)

    #odefunc.to(device)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    S0, I0, R0 = torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes,args.hidden))
    I0[0,list(args.I_indices)] = 1
    S0[0,:] = torch.ones((n_nodes,)) - I0[0,:]
    
    loss_all, val_loss_all = [], []
    best_loss = np.inf
    best_epoch = -1
    print("training...")
    for epoch in range(args.epochs):
        loss, val_loss, _, _, _ = train(model, optimizer, criterion, device, idx_train, idx_val, S_train, I_train, R_train, S_val, I_val, R_val, args.maxTime, args.deltaT, S0, I0, R0, A, args.model)
        #val_loss, _, _, _ = test(model, criterion, device, idx_val, S_val, I_val, R_val, maxTime, args.deltaT, S0, I0, R0)
        #scheduler.step(val_loss)

        loss_all.append(loss)
        #val_loss_all.append(val_loss)

        #print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))
        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))

        if val_loss <= best_loss:
            best_loss = val_loss
            s_test_time = time.time()
            test_loss, S_pred_t, I_pred_t, R_pred_t = test(model, criterion, device, idx_test, S_test, I_test, R_test, args.maxTime, args.deltaT, S0, I0, R0, A, args.model)
            e_test_time = time.time()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
    

    #S_rk, I_rk, R_rk = model.forward()
    '''
    s_test_time = time.time()
    test_loss, S_pred_t, I_pred_t, R_pred_t = test(model, criterion, device, idx_test, S_test, I_test, R_test, args.maxTime, args.deltaT, S0, I0, R0)
    e_test_time = time.time()'''

    print('Test Loss: {:.5f} at epoch: {:03d}'.format(test_loss, best_epoch))
    print('Test inference time: {:.5f}'.format(e_test_time - s_test_time))

    # baseline simple rk
    s_rk_time = time.time()
    I_sampled_t, S_sampled_t, R_sampled_t = runge_kutta_order4(sir, A, np.shape(A)[0], args.I_indices, args.beta, args.gamma, args.deltaT, args.maxTime)
    loss_S = mean_absolute_error(S_sampled_t, S_labels)
    loss_I = mean_absolute_error(I_sampled_t, I_labels)
    loss_R = mean_absolute_error(R_sampled_t, R_labels)
    loss_baseline = (loss_S + loss_I + loss_R)/3
    e_rk_time = time.time()
    print('Runge-kutta baseline Loss: {:.5f}'.format(loss_baseline))
    print('Time inference baseline: {:.5f}'.format(e_rk_time - s_rk_time))

    # baseline only for test
    loss_S = mean_absolute_error(get_labels_from_idx(S_sampled_t, idx_test), S_test)
    loss_I = mean_absolute_error(get_labels_from_idx(I_sampled_t, idx_test), I_test)
    loss_R = mean_absolute_error(get_labels_from_idx(R_sampled_t, idx_test), R_test)
    loss_baseline = (loss_S + loss_I + loss_R)/3
    print('Runge-kutta baseline test Loss: {:.5f}'.format(loss_baseline))

    # Keep results for all trials in csv
    save_trial_to_csv(args, best_epoch, test_loss.numpy(), loss_baseline, e_test_time - s_test_time, e_rk_time - s_rk_time)

    return 

if __name__ == '__main__':
    main()