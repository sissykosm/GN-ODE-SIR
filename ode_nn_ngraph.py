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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

import scipy
class ODEfunc(nn.Module):
    def __init__(self, A, beta, gamma, hidden1, device):
        super(ODEfunc, self).__init__()
        #self.A = torch.tensor(A, requires_grad=False).to(device)
        self.A = A
        self.beta = beta
        self.gamma = gamma
        #self.bn1 = nn.BatchNorm1d(hidden1)
        #self.bn2 = nn.BatchNorm1d(hidden1)
        #self.bn3 = nn.BatchNorm1d(hidden1)
        self.ln = nn.LayerNorm(hidden1)
        self.linear = nn.Linear(hidden1, hidden1)
        self.relu = nn.ReLU()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 1)

    def forward(self, t, x):
        sir_x = x[:int(x.size()[0]/4)*3,:]
        beta, gamma = x[int(x.size()[0]/4)*3:,0], x[int(x.size()[0]/4)*3:,1]
        
        sir_x = self.linear(sir_x)
        sir_x = self.relu(sir_x)

        S, I, R = sir_x[:int(x.size()[0]/4),:], sir_x[int(x.size()[0]/4):2*int(x.size()[0]/4),:], sir_x[2*int(x.size()[0]/4):,:]

        a = [self.A for _ in range(int((x.size()[0]/4)/np.shape(self.A)[0]))]
        bdiag = scipy.sparse.block_diag(a)
        idx = np.vstack((bdiag.row, bdiag.col))
        idx = torch.LongTensor(idx).to(x.device)
        
        AI = torch.zeros(I.size(), device=x.device).scatter_add_(0, idx[0,:].unsqueeze(1).repeat(1, I.size(1)), I[idx[1,:]])
        
        dS = -beta.unsqueeze(-1)*(torch.multiply(AI,S))
        dI = -dS - gamma.unsqueeze(-1)*I
        dR = gamma.unsqueeze(-1)*I
        '''
        bdiag = scipy.sparse.block_diag(a)
        bdiag = sparse_mx_to_torch_sparse_tensor(bdiag).to(x.device)
        
        
        dS = -beta.unsqueeze(-1)*(torch.multiply(torch.mm(bdiag,I),S))
        dI = beta.unsqueeze(-1)*(torch.multiply(torch.mm(bdiag,I),S)) - gamma.unsqueeze(-1)*I
        dR = gamma.unsqueeze(-1)*I
        '''
        # apply batch normalization 
        #dI = self.bn1(dI)
        #dS = self.bn2(dS)
        #dR = self.bn3(dR)

        # apply layer normalization 
        dS = self.ln(dS)
        dI = self.ln(dI)
        dR = self.ln(dR)
        return torch.cat((dS,dI,dR,x[int(x.size()[0]/4)*3:,:]))

# define neural ode solver
class ODEBlock(nn.Module):

    def __init__(self, maxTime, deltaT, n_nodes, indices, hidden1, odefunc, device):
        super(ODEBlock, self).__init__()
        self.maxTime = maxTime
        self.deltaT = deltaT
        self.device = device 

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
        #self.linearI1 = nn.Linear(1, hidden1)
        self.linearS1 = nn.Linear(1, hidden1)

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
        self.init_weights()


    def init_weights(self):
        self.linearS1.weight.data.normal_(0, 1)

    def forward(self, x):
        x = x.view(-1, x.size(2))
        S0, I0, R0, beta_gamma_batch = x[:,0], x[:,1], torch.zeros(x.size()[0], self.hidden1, device=self.device), x[:,3:]
        S = self.linearS1(torch.unsqueeze(S0, -1))
        I = self.linearS1(torch.unsqueeze(I0, -1))
        S = self.relu(S)
        I = self.relu(I)

        #print(R0.size()) #[batch_size*nodes] -->> [batch_size*nodes, hidden]
        #print(beta_gamma_batch.size()) #[batch_size*nodes, 2]
        #print(torch.cat((S,I,R0.cuda(),beta_gamma_batch)).size()) #[batch_size*nodes*4, hidden]

        sol = odeint(self.odefunc, torch.cat((S,I,R0,beta_gamma_batch)), self.integration_time, method='rk4')

        S, I, R = sol[:,:int(sol.size()[1]/4),:], sol[:,int(sol.size()[1]/4):2*int(sol.size()[1]/4),:], sol[:,2*int(sol.size()[1]/4):3*int(sol.size()[1]/4),:]

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

def train(model, optimizer, criterion, device, train_loader, val_loader, maxTime, deltaT, n_nodes):
    model.train()

    loss_all, val_loss_all = 0, 0
    items, val_items = 0, 0
    time_f = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        s_time = time.time()
        S_pred, I_pred, R_pred = model(x)
    
        e_time = time.time()
        time_f += e_time - s_time

        S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
        I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
        R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

        loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1).to(device), y.view(-1, y.size(2), y.size(3)))
        
        loss_all += loss.item()*3*S_pred_t.size()[0]*S_pred_t.size()[1]
        items += 3*S_pred_t.size()[0]*S_pred_t.size()[1]

        loss.backward()
        optimizer.step()

    model.eval()
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        S_pred, I_pred, R_pred = model(x)
        S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
        I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
        R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

        val_loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1).to(device), y.view(-1, y.size(2), y.size(3)))
        
        val_loss_all += val_loss.item()*3*S_pred_t.size()[0]*S_pred_t.size()[1]
        val_items += 3*S_pred_t.size()[0]*S_pred_t.size()[1]

    print("Time: ", time_f)
    return loss_all/items, val_loss_all/val_items
    #return 1, 1

def test(model, criterion, device, test_loader, maxTime, deltaT, n_nodes):
    model.eval()

    loss_all = 0
    items = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            S_pred, I_pred, R_pred = model(x)
            S_pred_t = get_sir_t_nodes_torch(torch.squeeze(S_pred), maxTime, deltaT, count=False)
            I_pred_t = get_sir_t_nodes_torch(torch.squeeze(I_pred), maxTime, deltaT, count=False)
            R_pred_t = get_sir_t_nodes_torch(torch.squeeze(R_pred), maxTime, deltaT, count=False)

            loss = criterion(torch.transpose(torch.cat((torch.unsqueeze(S_pred_t,-1), torch.unsqueeze(I_pred_t,-1), torch.unsqueeze(R_pred_t,-1)), -1), 0, 1).to(device), y.view(-1, y.size(2), y.size(3)))
            loss_all += loss.item()*3*S_pred_t.size()[0]*S_pred_t.size()[1]
            items += 3*S_pred_t.size()[0]*S_pred_t.size()[1]

    return loss_all/items
    #return 1

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
    args = parser.parse_args()

    #G, A = create_graph(50, './real_graphs/fb-social')
    G, A, A_dense = create_graph(50, args.dataset)
    #n_nodes = np.shape(A)[0]
    n_nodes = len(G.nodes())
    print(n_nodes)

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
        R0_per_I = torch.zeros((int(args.maxTime/args.deltaT),n_nodes,args.hidden), dtype=torch.float)
        I0_per_I[0,list(I_indices)] = 1
        S0_per_I[0,:] = torch.ones((n_nodes,), dtype=torch.float) - I0_per_I[0,:]
        S0.append(S0_per_I)
        I0.append(I0_per_I)
        R0.append(R0_per_I)

    return
    # create loaders for train, val, test (train includes 0.8 of the graph instances, val, test includes 0.2 each)

    labels = torch.transpose(torch.cat((torch.tensor(S_labels).unsqueeze(-1), torch.tensor(I_labels).unsqueeze(-1), torch.tensor(R_labels).unsqueeze(-1)), axis=-1), 1, 2)
    tensor_train_x, tensor_val_x, tensor_test_x = [], [], []
    tensor_train_y, tensor_val_y, tensor_test_y = [], [], []
    for i, I_indices in enumerate(args.I_indices): 
        beta_gamma_batch = torch.zeros(n_nodes, args.hidden, dtype=torch.float)
        beta_gamma_batch[:,0], beta_gamma_batch[:,1] = args.beta[i], args.gamma[i]
        if (i>=0) and (i <= int(args.train_val_test_ratio[0]*len(S_labels))-1):
            tensor_train_x.append(torch.cat((torch.unsqueeze(S0[i][0],1), torch.unsqueeze(I0[i][0],1), torch.unsqueeze(R0[i][0,:,0],1), beta_gamma_batch),-1))
            tensor_train_y.append(labels[i])
        elif (i > int(args.train_val_test_ratio[0]*len(S_labels))-1) and (i <= int((args.train_val_test_ratio[0]+args.train_val_test_ratio[1])*len(S_labels))-1):
            tensor_val_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1), beta_gamma_batch),-1))
            tensor_val_y.append(labels[i])
        else:
            tensor_test_x.append(torch.cat((torch.unsqueeze(S0[i][0],1),torch.unsqueeze(I0[i][0],1),torch.unsqueeze(R0[i][0,:,0],1), beta_gamma_batch),-1))
            tensor_test_y.append(labels[i])
    
   
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    print(device)

    if args.model == 'ode_nn':
        odefunc = ODEfunc(A, args.beta[0], args.gamma[0], args.hidden, device)
        model = ODEBlock(args.maxTime, args.deltaT, np.shape(A)[0], args.I_indices[0], args.hidden, odefunc, device)
    
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_all, val_loss_all = [], []
    best_loss = np.inf
    best_epoch = -1
    print("training...")
    for epoch in range(args.epochs):
        loss, val_loss = train(model, optimizer, criterion, device, train_loader, val_loader, args.maxTime, args.deltaT, n_nodes)
        #val_loss, _, _, _ = test(model, criterion, device, idx_val, S_val, I_val, R_val, maxTime, args.deltaT, S0, I0, R0)
        #scheduler.step(val_loss)

        loss_all.append(loss)
        #val_loss_all.append(val_loss)

        #print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))
        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            s_test_time = time.time()
            test_loss = test(model, criterion, device, test_loader, args.maxTime, args.deltaT, n_nodes)
            e_test_time = time.time()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    #print('Test Loss: {:.5f} at epoch: {:03d}'.format(test_loss, best_epoch))
    #print('Test inference time: {:.5f}'.format(e_test_time - s_test_time))

    if not path.exists(args.path_to_save + '/Metrics-trials-'+ os.path.relpath(args.dataset, './real_graphs/')):
        loss_baseline, rk_time = runge_kutta_baseline(A_dense, args, test_loader)
    else:
        loss_baseline, rk_time = 0, 0

    # Keep results for all trials in csv
    save_trial_to_csv(args, best_epoch, best_loss, test_loss, loss_baseline, e_test_time - s_test_time, rk_time)

    return 

if __name__ == '__main__':
    main()