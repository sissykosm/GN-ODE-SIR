import torch
import torch.nn as nn
#from torch_geometric.nn import GCNConv, GINConv

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
        for i in range(self.window-1):
            x = self.relu(self.mp[i+1](x, edge_index))
            x = self.dropout(x)
            lst.append(x.unsqueeze(1))
        x = torch.cat(lst, dim=1)
        #print(x.size())
        #print(x.size())
        #x = x.view(-1, self.window+1 ,x.size(1))
        #x = x[:,-1,:]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = torch.transpose(x, 0, 1)
        SIR_vectors = self.softmax(x)
        S, I, R = SIR_vectors.chunk(3, dim=-1)
        return S, I, R
        #return x.squeeze(1)
    
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
        for i in range(self.window-1):
            x = self.relu(self.mp[i+1](x, edge_index))
            x = self.dropout(x)
            lst.append(x.unsqueeze(1))
        x = torch.cat(lst, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = torch.transpose(x, 0, 1)
        SIR_vectors = self.softmax(x)
        S, I, R = SIR_vectors.chunk(3, dim=-1)
        return S, I, R
