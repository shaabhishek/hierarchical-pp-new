import numpy as np
import torch
import torch.nn as nn
from  torch.distributions.exponential import Exponential
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACD(nn.Module):
    def __init__(self,m =2 ):
        super().__init__()
        self.gamma =  nn.Parameter(torch.zeros(1))
        self.m = m
        self.alpha = nn.Parameter(torch.ones(self.m)/self.m)

    def forward(self, x, t, mask):
        #t : TxBSx2
        d_j = t[:,:,0]#TxBS
        batch_size, seq_length = x.size(1), x.size(0)
        past_influences = []
        for idx in range(self.m):
            t_pad = torch.cat([torch.zeros(idx+1,batch_size).to(device), d_j])[:-(idx+1), :] [:,:,None] #TxBSx1
            past_influences.append(t_pad)
        past_influences = torch.cat(past_influences, dim =-1) * self.alpha[None,None, :]#TxBSxm
        total_influence = torch.sum(past_influences, dim=-1) + self.gamma[None,:].exp()
        #To consider from time step 1
        m = Exponential(total_influence[1:,:])#T-1xBS
        ll_loss = (m.log_prob(d_j[1:, :])
                    ).sum()
        
        metric_dict = {'true_ll': -ll_loss.detach(), "marker_acc":0., "marker_acc_count":1.}
        with torch.no_grad():
            time_mse = torch.abs(d_j[1:,:]- 1./total_influence[1:,:]) * mask[1:, :]
            metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
            metric_dict['time_mse_count'] = mask[1:,:].sum().detach().cpu().numpy()
        return -ll_loss, metric_dict            


        

        


