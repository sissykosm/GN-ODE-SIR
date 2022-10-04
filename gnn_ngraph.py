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

from torch_geometric.data import Data, DataLoader

import ndlib.models.ModelConfig as mc
import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm

#from models import GCN, GIN
from ode_nn import sir_nx, sir_pandas, sir, runge_kutta_order4, get_sir_t_nodes, get_sir_t_nodes_torch, csv_trials, save_trial_to_csv, create_graph, get_train_test, get_labels_from_idx

#torch.manual_seed(0)
#np.random.seed(0)

from torch_geometric.nn import GCNConv, GINConv

#### GCN model ####
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, penultimate_dim, n_targets, dropout, window):
        super(GCN, self).__init__()
        self.window = window
        mp_list = [GCNConv(input_dim, hidden_dim)]
        #fc1_list = [nn.Linear(hidden_dim*window, penultimate_dim)]
        #fc2_list = [nn.Linear(penultimate_dim, n_targets)]
        for i in range(window-1):
            mp_list.append(GCNConv(hidden_dim, hidden_dim))
            #fc1_list.append(nn.Linear(hidden_dim*window, penultimate_dim))
            #fc2_list.append(nn.Linear(penultimate_dim, n_targets))
        self.mp = nn.ModuleList(mp_list)
        #self.fc1 = nn.ModuleList(fc1_list)
        #self.fc2 = nn.ModuleList(fc2_list)
        self.fc1 = nn.Linear(hidden_dim, penultimate_dim)
        self.fc2 = nn.Linear(penultimate_dim, n_targets)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_features, edge_index):
        lst = list()
        x = self.relu(self.mp[0](x_features, edge_index))
        x = self.dropout(x)
        lst.append(x.unsqueeze(1))
        for i in range(self.window-2):
            x = self.relu(self.mp[i+1](x, edge_index))
            x = self.dropout(x)
            lst.append(x.unsqueeze(1))
        x = torch.cat(lst, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = torch.transpose(x, 0, 1)
        SIR_vectors = self.softmax(x)
        return SIR_vectors
    
#### GIN model ####
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, penultimate_dim, n_targets, dropout, window):
        super(GIN, self).__init__()
        self.window = window
        mp_list = [GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))]
        
        for i in range(window-1):
            mp_list.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim))))
    
        self.mp = nn.ModuleList(mp_list)
        self.fc1 = nn.Linear(hidden_dim, penultimate_dim)
        self.fc2 = nn.Linear(penultimate_dim, n_targets)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_features, edge_index):
        lst = list()
        x = self.relu(self.mp[0](x_features, edge_index))
        x = self.dropout(x)
        lst.append(x.unsqueeze(1))
        for i in range(self.window-2):
            x = self.relu(self.mp[i+1](x, edge_index))
            x = self.dropout(x)
            lst.append(x.unsqueeze(1))
        x = torch.cat(lst, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = torch.transpose(x, 0, 1)
        SIR_vectors = self.softmax(x)
        return SIR_vectors

def load_SIR_labels(dataset, path_to_save, G, I_indices, beta, gamma, sim, maxTime):
    if os.path.exists(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + '.pkl'):
        S_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        I_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
        R_labels = pickle.load(open(path_to_save + '/' + dataset[14:] + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "rb"))
    else:
        #S_per_sim, I_per_sim, R_per_sim, _, _ = sir_nx(G, I_indices, beta, gamma, sim, maxTime)
        S_per_sim, I_per_sim, R_per_sim = sir_pandas(G, I_indices, beta, gamma, sim, maxTime)
        
        S_labels, I_labels, R_labels = np.mean(S_per_sim,0), np.mean(I_per_sim,0), np.mean(R_per_sim,0)
        pickle.dump(S_labels, open(path_to_save + '/' + dataset[14:] + '-S-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))
        pickle.dump(I_labels, open(path_to_save + '/' + dataset[14:] + '-I-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))
        pickle.dump(R_labels, open(path_to_save + '/' + dataset[14:] + '-R-' + '-'.join(str(i) for i in I_indices) + ".pkl", "wb" ))

    return S_labels, I_labels, R_labels

def train(model, optimizer, criterion, device, train_loader, val_loader, maxTime, deltaT):
    model.train()

    loss_all, val_loss_all = 0, 0
    items, val_items = 0, 0
    time_f = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        s_time = time.time()
        y_pred = model(data.x, data.edge_index)
        e_time = time.time()
    
        time_f += e_time - s_time

        loss = criterion(torch.transpose(y_pred, 0, 1), data.y[:,1:,:])
        loss.backward()
        optimizer.step()

        loss_all += loss.item()*3*data.y.size()[0]*(data.y.size()[1]-1)
        items += 3*data.y.size()[0]*(data.y.size()[1]-1)

    model.eval()
    for data in val_loader:
        data = data.to(device)
        y_pred = model(data.x, data.edge_index)
        val_loss = criterion(torch.transpose(y_pred, 0, 1), data.y[:,1:,:])

        val_loss_all += val_loss.item()*3*data.y.size()[0]*(data.y.size()[1]-1)
        val_items += 3*data.y.size()[0]*(data.y.size()[1]-1)

    print("Time: ", time_f)
    return loss_all/items, val_loss_all/val_items

def test(model, criterion, device, test_loader, maxTime, deltaT):
    model.eval()

    loss_all = 0
    items = 0

    test_loss_all = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            y_pred = model(data.x, data.edge_index)
            loss = criterion(torch.transpose(y_pred, 0, 1), data.y[:,1:,:])

            loss_all += loss.item()*3*data.y.size()[0]*(data.y.size()[1]-1)
            items += 3*data.y.size()[0]*(data.y.size()[1]-1)

            test_loss_all.append(loss.item())
    
    return loss_all/items, test_loss_all

def runge_kutta_baseline(A, args, test_loader):
    # baseline simple rk
    s_rk_time = time.time()
    loss = []

    I_indices_test = args.I_indices[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
    beta_test, gamma_test = args.beta[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):], args.gamma[int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(args.I_indices)):]
    for i, data in enumerate(test_loader): 
        I_sampled_t, S_sampled_t, R_sampled_t = runge_kutta_order4(sir, A, np.shape(A)[0], I_indices_test[i], beta_test[i], gamma_test[i], args.deltaT, args.maxTime)

        loss_S = mean_absolute_error(S_sampled_t, torch.transpose(data.y, 0, 1).cpu().numpy()[:,:,0])
        loss_I = mean_absolute_error(I_sampled_t, torch.transpose(data.y, 0, 1).cpu().numpy()[:,:,1])
        loss_R = mean_absolute_error(R_sampled_t, torch.transpose(data.y, 0, 1).cpu().numpy()[:,:,2])
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
    G, A, A_dense = create_graph(50, args.dataset)
    n_nodes = np.shape(A)[0]

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
        S0_per_I, I0_per_I, R0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes))
        I0_per_I[0,list(I_indices)] = 1
        S0_per_I[0,:] = torch.ones((n_nodes,)) - I0_per_I[0,:]
        S0.append(S0_per_I)
        I0.append(I0_per_I)
        R0.append(R0_per_I)


    # create loaders for train, val, test (train includes 0.8 of the graph instances, val, test includes 0.2 each)
    print('ok')
    labels = torch.transpose(torch.cat((torch.tensor(S_labels).unsqueeze(-1), torch.tensor(I_labels).unsqueeze(-1), torch.tensor(R_labels).unsqueeze(-1)), axis=-1), 1, 2)
    data_list_train, data_list_val, data_list_test = [], [], []
    if args.out_of_dist==False:
        for i, I_indices in enumerate(args.I_indices): 
            beta_gamma_batch = torch.zeros(n_nodes, 2, dtype=torch.float)
            beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
            if (i>=0) and (i <= int(args.train_val_test_ratio[0]*len(S_labels))-1):
                #data_list_train.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
                data_list_train.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            elif (i > int(args.train_val_test_ratio[0]*len(S_labels))-1) and (i <= int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(S_labels))-1):
                #data_list_val.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
                data_list_val.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            else:
                #data_list_test.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
                data_list_test.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
    else:
        dict_idx = pickle.load(open(args.path_to_save + '/out-of-dist-gamma.pkl', "rb"))
        #count_val = 0
        idx_test = dict_idx['test']
        for i, I_indices in enumerate(args.I_indices): 
            beta_gamma_batch = torch.zeros(n_nodes, 2, dtype=torch.float)
            beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
            if i in dict_idx['train']:
                data_list_train.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            elif i in dict_idx['val']:
                data_list_val.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            else:
                data_list_test.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0],1),beta_gamma_batch),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
    
    train_loader = DataLoader(data_list_train, shuffle = True, batch_size=args.batch_size)
    val_loader = DataLoader(data_list_val, shuffle = False, batch_size=args.batch_size)
    test_loader = DataLoader(data_list_test, shuffle = False)

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    if args.model == 'GCN':
        model = GCN(5, args.hidden, int(args.hidden/2), 3, 0.1, args.maxTime)
    else:
        model = GIN(5, args.hidden, int(args.hidden/2), 3, 0.1, args.maxTime)
    
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_all, val_loss_all = [], []
    best_loss = np.inf
    best_epoch = -1
    print("training...")
    for epoch in range(args.epochs):
        loss, val_loss = train(model, optimizer, criterion, device, train_loader, val_loader, args.maxTime, args.deltaT)
        #val_loss, _, _, _ = test(model, criterion, device, idx_val, S_val, I_val, R_val, maxTime, args.deltaT, S0, I0, R0)
        #scheduler.step(val_loss)

        loss_all.append(loss)
        #val_loss_all.append(val_loss)

        #print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))
        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            s_test_time = time.time()
            test_loss, test_loss_all = test(model, criterion, device, test_loader, args.maxTime, args.deltaT)
            e_test_time = time.time()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    #print('Test Loss: {:.5f} at epoch: {:03d}'.format(test_loss, best_epoch))
    #print('Test inference time: {:.5f}'.format(e_test_time - s_test_time))

    if not path.exists(args.path_to_save + '/Metrics-trials2-'+ os.path.relpath(args.dataset, './real_graphs/')):
        #loss_baseline, rk_time = runge_kutta_baseline(A_dense, args, test_loader)
        loss_baseline, rk_time = 0, 0
    else:
        loss_baseline, rk_time = 0, 0

    # Keep results for all trials in csv
    if not args.out_of_dist:
        save_trial_to_csv(args, best_epoch, best_loss, test_loss, loss_baseline, e_test_time - s_test_time, rk_time)
    else:
        csv_trials(args.path_to_save + '/Out-of-dist-gamma-'+ os.path.relpath(args.dataset, './real_graphs/'), [str(i) for i in idx_test], test_loss_all)
        list_to_csv = [args.trial, args.model, args.lr, args.epochs, args.deltaT, args.maxTime, args.hidden, best_epoch, best_loss, test_loss, e_test_time - s_test_time]
        csv_trials(args.path_to_save + '/Out-of-dist-gamma-trials-'+ os.path.relpath(args.dataset, './real_graphs/'), ["trial", "model", "lr", "epochs", "deltaT", "maxTime", "hidden", "best_epoch", "val_loss", "test_loss", "n_ode_time"], list_to_csv)
    
    return 

if __name__ == '__main__':
    main()
    
'''
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
        S0_per_I, I0_per_I, R0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes)), torch.zeros((int(args.maxTime/args.deltaT),n_nodes,args.hidden))
        I0_per_I[0,list(I_indices)] = 1
        S0_per_I[0,:] = torch.ones((n_nodes,)) - I0_per_I[0,:]
        S0.append(S0_per_I)
        I0.append(I0_per_I)
        R0.append(R0_per_I)


    # create loaders for train, val, test (train includes 0.8 of the graph instances, val, test includes 0.2 each)

    labels = torch.transpose(torch.cat((torch.tensor(S_labels).unsqueeze(-1), torch.tensor(I_labels).unsqueeze(-1), torch.tensor(R_labels).unsqueeze(-1)), axis=-1), 1, 2)
    data_list_train, data_list_val, data_list_test = [], [], []
    for i, I_indices in enumerate(args.I_indices): 
        beta_gamma_batch = torch.zeros(n_nodes, 2, dtype=torch.float)
        beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
        if (i>=0) and (i <= int(args.train_val_test_ratio[0]*len(S_labels))-1):
            #data_list_train.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            data_list_train.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
        elif (i > int(args.train_val_test_ratio[0]*len(S_labels))-1) and (i <= int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(S_labels))-1):
            #data_list_val.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            data_list_val.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
        else:
            #data_list_test.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1)),-1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
            data_list_test.append(Data(x = torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1),beta_gamma_batch), -1), edge_index = torch.LongTensor(np.array(np.nonzero(A))), y = labels[i]))
'''