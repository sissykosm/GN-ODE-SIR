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
from ode_nn import sir_nx, sir_pandas, sir_torch, sir, runge_kutta_order4, get_sir_t_nodes, get_sir_t_nodes_torch, csv_trials, save_trial_to_csv, get_train_test, get_labels_from_idx

#torch.manual_seed(0)
#np.random.seed(0)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

import scipy
class ODEfunc(nn.Module):
    def __init__(self, A_list, hidden1, device):
        super(ODEfunc, self).__init__()
        #self.A = torch.tensor(A, requires_grad=False).to(device)
        self.A_list = A_list
        #self.bn1 = nn.BatchNorm1d(hidden1)
        #self.bn2 = nn.BatchNorm1d(hidden1)
        #self.bn3 = nn.BatchNorm1d(hidden1)
        self.ln = nn.LayerNorm(hidden1)
        self.linear = nn.Linear(hidden1, hidden1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 1)

    def forward(self, t, x):
        beta, gamma, graph_idx = x[3,:,0], x[3,:,1], x[3,:,2]
        
        sir_x = x[:3,:,:]
        sir_x = sir_x.view(-1,x.size(-1))
        sir_x = self.linear(sir_x)
        sir_x = self.sigmoid(sir_x)
        sir_x = sir_x.view(3,-1,sir_x.size(-1))

        S, I, R = sir_x[0,:,:], sir_x[1,:,:], sir_x[2,:,:]

        a = []
        for i in torch.nonzero(graph_idx).flatten():
            a.append(self.A_list[graph_idx[i].type(torch.int64)-1])

        bdiag = scipy.sparse.block_diag(a)
        idx = np.vstack((bdiag.row, bdiag.col))
        idx = torch.LongTensor(idx).to(x.device)
        
        AI = torch.zeros(I.size(), device=x.device).scatter_add_(0, idx[0,:].unsqueeze(1).repeat(1, I.size(1)), I[idx[1,:]])
        
        dS = -beta.unsqueeze(-1)*(torch.multiply(AI,S))
        dI = -dS - gamma.unsqueeze(-1)*I
        dR = gamma.unsqueeze(-1)*I

        # apply layer normalization 
        #dS = self.ln(dS)
        #dI = self.ln(dI)
        #dR = self.ln(dR)
        return torch.stack((dS,dI,dR,torch.zeros_like(x[3,:,:])))

# define neural ode solver
class ODEBlock(nn.Module):

    def __init__(self, maxTime, deltaT, hidden1, odefunc, device):
        super(ODEBlock, self).__init__()
        self.maxTime = maxTime
        self.deltaT = deltaT
        self.device = device 

        # SIR parameters
        self.maxTime = maxTime
        self.deltaT = deltaT
        self.integration_time = torch.from_numpy(np.arange(0, self.maxTime, self.deltaT)).to(device)
        self.odefunc = odefunc

        # model parameters

        # input linear layers
        self.hidden1 = hidden1
        #self.linearI1 = nn.Linear(1, hidden1)
        self.linearS1 = nn.Linear(1, hidden1)
        self.ln = nn.LayerNorm(hidden1)
        # more linear layers
        self.linear3 = nn.Linear(hidden1, 4)
        self.relu3 = nn.ReLU()

        # output linear layers 
        #self.linearI2 = nn.Linear(hidden1,1)
        self.linearS2 = nn.Linear(4,1)
        #self.linearR2 = nn.Linear(hidden1,1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        #self.init_weights()


    def init_weights(self):
        self.linearS1.weight.data.normal_(0, 1)

    def forward(self, x):
        S0, I0, R0, beta_gamma_batch = x[:,0], x[:,1], x[:,2], x[:,3:]
        S = self.linearS1(torch.unsqueeze(S0, -1))
        I = self.linearS1(torch.unsqueeze(I0, -1))
        R = self.linearS1(torch.unsqueeze(R0, -1))        
        S = self.relu(S)
        I = self.relu(I)
        R = self.relu(R)
        
        #print(R0.size()) #[batch_size*nodes] -->> [batch_size*nodes, hidden]
        #print(beta_gamma_batch.size()) #[batch_size*nodes, 2]
        #print(torch.cat((S,I,R0.cuda(),beta_gamma_batch)).size()) #[batch_size*nodes*4, hidden]

        sol = odeint(self.odefunc, torch.stack((S,I,R,beta_gamma_batch)), self.integration_time, method='euler')
        S, I, R = sol[:,0,:,:], sol[:,1,:,:], sol[:,2,:,:]

        S = self.linear3(S)
        I = self.linear3(I)
        R = self.linear3(R) 

        S = self.linearS2(self.relu3(S))
        I = self.linearS2(self.relu3(I))
        R = self.linearS2(self.relu3(R)) 
        
        SIR_vectors = torch.cat((S, I, R), -1)
        
        SIR_vectors = self.softmax(SIR_vectors)
        S, I, R = SIR_vectors.chunk(3, dim=-1)
        return S, I, R

def create_graphs(graph_label='none'):
    A_list = []
    if graph_label != 'none':
        for graph in graph_label[14:].split('+'):
            G = pickle.load(open(graph_label[:14]+ graph + ".pkl", "rb"))
            G = G.to_undirected()
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            print('nodes', len(G.nodes()))
            print('edges', len(G.edges()))
            A_list.append(nx.adjacency_matrix(G))
    return A_list

def load_SIR_labels(dataset, path_to_save, I_indices, sim):
    if dataset == 'wiki-vote': # or dataset == 'enron':
        S_labels = pickle.load(open(path_to_save + '/' + dataset + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))/sim
        I_labels = pickle.load(open(path_to_save + '/' + dataset + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))/sim
        R_labels = pickle.load(open(path_to_save + '/' + dataset + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))/sim
    else:
        S_labels = pickle.load(open(path_to_save + '/' + dataset + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        I_labels = pickle.load(open(path_to_save + '/' + dataset + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        R_labels = pickle.load(open(path_to_save + '/' + dataset + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))

    return S_labels, I_labels, R_labels

def loader(N, batch_size, tensor_x, tensor_y, train=False):
    if train:
        idx = np.random.permutation(N)
    else:
        idx = np.arange(N)

    x_batches = []
    y_batches = []
    for i in range(0, N, batch_size):
        x_temp = [tensor_x[idx[j]] for j in range(i, min(i+batch_size, N))]
        x_temp = torch.cat(x_temp, dim=0)

        y_temp = [tensor_y[idx[j]] for j in range(i, min(i+batch_size, N))]
        y_temp = torch.cat(y_temp, dim=0)

        x_batches.append(x_temp)
        y_batches.append(y_temp)
    return x_batches, y_batches

def train(model, optimizer, criterion, device, x_train, y_train, x_val, y_val, maxTime, deltaT):
    model.train()

    loss_all, val_loss_all = 0, 0
    items, val_items = 0, 0
    time_f = 0
    for i in range(len(x_train)):
        x = x_train[i].to(device)
        y = y_train[i].to(device)
        optimizer.zero_grad()

        s_time = time.time()
        S_pred, I_pred, R_pred = model(x)
        
        e_time = time.time()
        time_f += e_time - s_time

        S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
        I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
        R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

        loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1)[:,1:,:].to(device), y[:,1:,:])
        
        loss_all += loss.item()*3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]
        items += 3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]

        loss.backward()
        optimizer.step()

    model.eval()
    for i in range(len(x_val)):
        x = x_val[i].to(device)
        y = y_val[i].to(device)
        S_pred, I_pred, R_pred = model(x)
        S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
        I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
        R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

        val_loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1)[:,1:,:].to(device), y[:,1:,:])
        
        val_loss_all += val_loss.item()*3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]
        val_items += 3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]

    print("Time: ", time_f)
    return loss_all/items, val_loss_all/val_items

def test(model, criterion, device, x_test, y_test, maxTime, deltaT):
    model.eval()

    loss_all = 0
    items = 0

    with torch.no_grad():
        for i in range(len(x_test)):
            x = x_test[i].to(device)
            y = y_test[i].to(device)

            S_pred, I_pred, R_pred = model(x)
            S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
            I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
            R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

            loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1)[:,1:,:].to(device), y[:,1:,:])
            loss_all += loss.item()*3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]
            items += 3*(S_pred_t.size()[0]-1)*S_pred_t.size()[1]

    return loss_all/items

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
    parser.add_argument('--deltaT', type=float, default=0.5, metavar='N', help='deltaT for SIR')
    parser.add_argument('--maxTime', type=int, default=20, metavar='N', help='maxTime for SIR')
    parser.add_argument('--hidden', type=int, default=32, metavar='N', help='Size of hidden layer of NN')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size')
    parser.add_argument("--path_to_save", help="Path to save plots", default="./plots")
    parser.add_argument('--trial', type=int, default=32, metavar='N', help='Number of trial')
    parser.add_argument("--dataset", help="Path to the dataset", default="none")
    parser.add_argument("--train_val_test_ratio", help="Ratio for train, val and test split from the initial time series", nargs=3, type=float, default=[5e-1, 1e-1, 4e-1])
    parser.add_argument('--model', default='ode_nn', help="model to train, choose between lstm, tcn, encdecgru", type=str)
    args = parser.parse_args()

    A_list = create_graphs(args.dataset)
    print(len(A_list))
    instances_per_graph = [36, 36, 36, 36, 36, 120]
    n_graphs = len(instances_per_graph)-2
    val_len = int(instances_per_graph[-1]/2)

    count_val = 0
    tensor_train_x, tensor_val_x, tensor_test_x = [], [], []
    tensor_train_y, tensor_val_y, tensor_test_y = [], [], []
    for graph_idx, graph in enumerate(args.dataset[14:].split('+')):
        if graph=='wiki-vote':
            path_load = './multi-graph-1/Experiments-gpu-seed2'
        elif graph=='enron':
            path_load = './multi-graph-1/Experiments2-seed2'
        else:
            path_load = './multi-graph-1/'+args.path_to_save.split('/')[-1].split('-')[0]+'-'+args.path_to_save.split('/')[-1].split('-')[1]
        
        I_indices = pickle.load(open(path_load+'-'+ graph + '/' + "initial-seed.pkl", "rb"))[:instances_per_graph[graph_idx]] #set 60
        betas = pickle.load(open(path_load+'-'+ graph + '/' + "initial-beta.pkl", "rb"))[:instances_per_graph[graph_idx]]
        gammas = pickle.load(open(path_load+'-'+ graph + '/' + "initial-gamma.pkl", "rb"))[:instances_per_graph[graph_idx]]
        n_nodes = np.shape(A_list[graph_idx])[0]

        for i, indices in enumerate(I_indices):
            beta_gamma_batch = torch.zeros(n_nodes, args.hidden, dtype=torch.float)
            beta_gamma_batch[:,0], beta_gamma_batch[:,1], beta_gamma_batch[0,2] = betas[i], gammas[i], graph_idx+1
            #print(beta_gamma_batch[0,2])

            # create labels for all nodes per t using monte-carlo simulations
            S_mc, I_mc, R_mc = load_SIR_labels(graph, path_load+'-'+ graph, indices, args.sim)
            
            # initial condition per seed
            S0_per_I = torch.zeros(1, n_nodes, dtype=torch.float)
            I0_per_I = torch.zeros(1, n_nodes, dtype=torch.float)
            R0_per_I = torch.zeros(1, n_nodes, dtype=torch.float)
            I0_per_I[0,list(I_indices)] = 1
            S0_per_I[0,:] = torch.ones((n_nodes,), dtype=torch.float) - I0_per_I[0,:]
            
            S0_per_I, I0_per_I, R0_per_I = torch.squeeze(S0_per_I), torch.squeeze(I0_per_I), torch.squeeze(R0_per_I)
            if graph_idx<=n_graphs:
                tensor_train_x.append(torch.cat((torch.unsqueeze(torch.squeeze(S0_per_I),-1), torch.unsqueeze(torch.squeeze(I0_per_I),-1), torch.unsqueeze(torch.squeeze(R0_per_I),-1), beta_gamma_batch),-1))
                tensor_train_y.append(torch.transpose(torch.cat((torch.tensor(S_mc).unsqueeze(-1), torch.tensor(I_mc).unsqueeze(-1), torch.tensor(R_mc).unsqueeze(-1)), axis=-1), 0, 1))
            elif count_val<val_len:
                tensor_val_x.append(torch.cat((torch.unsqueeze(torch.squeeze(S0_per_I),-1), torch.unsqueeze(torch.squeeze(I0_per_I),-1), torch.unsqueeze(torch.squeeze(R0_per_I),-1), beta_gamma_batch),-1))
                tensor_val_y.append(torch.transpose(torch.cat((torch.tensor(S_mc).unsqueeze(-1), torch.tensor(I_mc).unsqueeze(-1), torch.tensor(R_mc).unsqueeze(-1)), axis=-1), 0, 1))
                count_val+=1
            else:
                tensor_test_x.append(torch.cat((torch.unsqueeze(torch.squeeze(S0_per_I),-1), torch.unsqueeze(torch.squeeze(I0_per_I),-1), torch.unsqueeze(torch.squeeze(R0_per_I),-1), beta_gamma_batch),-1))
                tensor_test_y.append(torch.transpose(torch.cat((torch.tensor(S_mc).unsqueeze(-1), torch.tensor(I_mc).unsqueeze(-1), torch.tensor(R_mc).unsqueeze(-1)), axis=-1), 0, 1))

        
    # create loaders for train, val, test (train includes 0.8 of the graph instances, val, test includes 0.2 each)
    x_batches_train, y_batches_train = loader(sum(instances_per_graph[:-1]), args.batch_size, tensor_train_x, tensor_train_y, True)
    x_batches_val, y_batches_val = loader(val_len, args.batch_size, tensor_val_x, tensor_val_y)
    x_batches_test, y_batches_test = loader(val_len, args.batch_size, tensor_test_x, tensor_test_y)

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    print(device)

    if args.model == 'ode_nn':
        odefunc = ODEfunc(A_list, args.hidden, device)
        model = ODEBlock(args.maxTime, args.deltaT, args.hidden, odefunc, device)
    
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_all, val_loss_all = [], []
    best_loss = np.inf
    best_epoch = -1
    print("training...")
    for epoch in range(args.epochs):
        loss, val_loss = train(model, optimizer, criterion, device, x_batches_train, y_batches_train, x_batches_val, y_batches_val, args.maxTime, args.deltaT)
        #val_loss, _, _, _ = test(model, criterion, device, idx_val, S_val, I_val, R_val, maxTime, args.deltaT, S0, I0, R0)
        #scheduler.step(val_loss)

        loss_all.append(loss)
        #val_loss_all.append(val_loss)

        #print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))
        print('Epoch: {:03d}, Train Loss: {:.10f}, Val Loss: {:.10f}'.format(epoch, loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            s_test_time = time.time()
            test_loss = test(model, criterion, device, x_batches_test, y_batches_test, args.maxTime, args.deltaT)
            e_test_time = time.time()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    #print('Test Loss: {:.5f} at epoch: {:03d}'.format(test_loss, best_epoch))
    #print('Test inference time: {:.5f}'.format(e_test_time - s_test_time))

    if not path.exists(args.path_to_save + '/Metrics-trials-'+ os.path.relpath(args.dataset, './real_graphs/')):
        #loss_baseline, rk_time = runge_kutta_baseline(A_dense, args, test_loader)
        loss_baseline, rk_time = 0, 0
    else:
        loss_baseline, rk_time = 0, 0

    # Keep results for all trials in csv
    list_to_csv = [args.trial, args.model, args.lr, args.epochs, args.deltaT, args.maxTime, args.hidden, best_epoch, best_loss, test_loss, e_test_time - s_test_time] #loss_baseline, rk_time]
    csv_trials(args.path_to_save + '/Metrics-trials-'+ os.path.relpath(args.dataset, './real_graphs/'), ["trial", "model", "lr", "epochs", "deltaT", "maxTime", "hidden", "best_epoch", "val_loss", "test_loss", "n_ode_time"], list_to_csv)
    
    return 

if __name__ == '__main__':
    main()