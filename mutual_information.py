import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import pandas as pd

# Main function: Trains the MINE and the computes the MI
def mu_info(data, S,mine_net):
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
    results = train(data, S, mine_net,mine_net_optim)
    joint, marginal = sample_batch(data,S,batch_size=100)
    result,temp0,temp1 = mutual_information(joint, marginal,mine_net)
    return result,results

# MINE network
class Mine(nn.Module):
    def __init__(self, data_size, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(data_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, data):
        data = data.float()
        output = F.elu(self.fc1(data))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

# Compute mutual information
def mutual_information(joint, marginal, mine_net):    
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

# Train MINE network
def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    joint, marginal = batch
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    mine_net_optim.zero_grad()
    loss.backward(retain_graph = True)
    mine_net_optim.step()
    return mi_lb

# Train function
def train(data, S, mine_net,mine_net_optim, batch_size = 100, iter_num=int(1e+1)):
    ma_et = 1
    results = []
    for i in range(iter_num):
        joint, marginal = sample_batch(data,S,batch_size)
        batch = joint,marginal
        mi_lb = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        results.append(mi_lb.detach().cpu().numpy())
        if (i+1)%(1000)==0:
            print(results[-1])
    result = mi_lb
    return results

# Generate joint and marginal distribuition between data and the sensitive feature
def sample_batch(data, S, batch_size=100):
    if type(S) is np.ndarray and type(data) == np.ndarray: 
        S = torch.from_numpy(S).to(device='cuda:0')
        data = torch.from_numpy(data).to(device='cuda:0')
        
    index = torch.randperm(S.shape[0])
    joint_idx = torch.randperm(S.shape[0])
    marginal_idx = torch.randperm(S.shape[0])

    data_bar = data[joint_idx,:]
    S_bar = S[marginal_idx,:]
    marginal = torch.cat((data_bar,S_bar),1)
    joint = torch.cat((data[index,:],S[index,:]),1)
    
    batch = joint, marginal
    return batch
