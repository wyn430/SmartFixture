'''
@author: yinanw
Actor-Critic Network
'''


import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transform as T
import torch.optim as optim
torch.manual_seed(24)
random.seed(24)
np.random.seed(24)

def knn(x, k):
    
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, features, device, k=20, idx=None):
    
    batch_size = features.size(0)
    num_points = features.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = features.size()
    
    #print(features.shape)
    features = features.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    output_feature = features.view(batch_size*num_points, -1)[idx, :]
    output_feature = output_feature.view(batch_size, num_points, k, num_dims) 
    features = features.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    output_feature = torch.cat((output_feature-features, features), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return output_feature      # (batch_size, 2*num_dims, num_points, k)


class Shared_Module_GCN(nn.Module):
    def __init__(self, feature_dims, k, emb_dims, device):
        super(Shared_Module_GCN, self).__init__()
        
        self.k = k
        self.emb_dims = emb_dims
        self.feature_dims = feature_dims
        self.device = device
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.feature_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.Tanh())
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.Tanh())
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.Tanh())
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.Tanh())
        

    def forward(self, x, features):
        ### x stores the coordinates of all the points
        ### features stores the stress, displacement, etc.
        x = x.permute(0,2,1)
        features = features.permute(0,2,1)
        batch_size = x.size(0)
        # (batch_size, num_features, num_points) -> (batch_size, num_features*2, num_points, k)
        features = get_graph_feature(x, features, self.device, k=self.k)
        # (batch_size, num_features*2, num_points, k) -> (batch_size, 64, num_points, k)
        features = self.conv1(features)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        features1 = features.max(dim=-1, keepdim=False)[0]    

        features = get_graph_feature(x, features1, self.device, k=self.k)     
        features = self.conv2(features)                       
        features2 = features.max(dim=-1, keepdim=False)[0]    

        features = get_graph_feature(x, features2, self.device, k=self.k) 
        features = self.conv3(features)                       
        features3 = features.max(dim=-1, keepdim=False)[0]
        
        features = get_graph_feature(x, features3, self.device, k=self.k) 
        features = self.conv4(features)                       
        features4 = features.max(dim=-1, keepdim=False)[0]    

        features = torch.cat((features1, features2, features3, features4), dim=1)  

        features = self.conv5(features)                      # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        
        
        features1 = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        features2 = F.adaptive_avg_pool1d(features, 1).view(batch_size, -1)        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        features = torch.cat((features1, features2), 1)              # (batch_size, emb_dims*2)
        
        
        
        return features
    
    
class Actor2_Arm_Conv(nn.Module):
    def __init__(self, shared_output_dim, arm_list, action2_dim):
        
        super(Actor2_Arm_Conv, self).__init__()
        
        self.input_dim = shared_output_dim
        self.arm_list = arm_list
        self.action2_dim = action2_dim
        
        layer_list = []
        
        for l in range(len(arm_list)):
            if l == 0:
                layer_list.append(nn.Conv1d(self.input_dim, self.arm_list[l], kernel_size=1, bias=False))
                layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.ReLU())
            else:
                layer_list.append(nn.Conv1d(self.arm_list[l-1], self.arm_list[l], kernel_size=1, bias=False))
                layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.ReLU())
        
        
        self.Act2 = nn.Sequential(*layer_list)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask):
        
        batch_size = x.size(0)
        output = self.Act2(x)
        output = output.permute(0,2,1) #(batch_size, emb_dims, num_points) --> (batch_size, num_points, emb_dims)
        #print(output.shape)
        output = F.adaptive_max_pool1d(output, 1).view(batch_size, -1) #(batch_size, num_points, emb_dims) --> (batch_size, num_points)
        #print(output.shape)
        #output = torch.mul(output, mask)
        
        output = self.softmax(output)
        assert output.size(1) == self.action2_dim ##confirm the dimension
        
        output = torch.mul(output, mask)
        
        
        return output
    
    
class Critic_Arm_new(nn.Module):
    def __init__(self, shared_output_dim, arm_list):
        
        super(Critic_Arm_new, self).__init__()
        
        self.input_dim = shared_output_dim
        self.arm_list = arm_list
        
        layer_list = []
        
        for l in range(len(arm_list)):
            if l == 0:
                layer_list.append(nn.Linear(self.input_dim, self.arm_list[l]))
                layer_list.append(nn.ReLU())
            else:
                layer_list.append(nn.Linear(self.arm_list[l-1], self.arm_list[l]))
                layer_list.append(nn.ReLU())
            
        layer_list.append(nn.Linear(self.arm_list[-1], 1))
        layer_list.append(nn.Sigmoid())
        
        self.Critic = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        
        batch_size = x.size(0)
        features1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        features2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        features = torch.cat((features1, features2), 1)              # (batch_size, emb_dims*2)
        output = self.Critic(features)
        return output


class Shared_Module(nn.Module):
    def __init__(self, channel_list):
        
        super(Shared_Module, self).__init__()
        
        self.state_dim = state_dim
        self.channel_list = channel_list
        
        layer_list = []
        
        for l in range(len(channel_list)-1):
            layer_list.append(nn.Conv2d(channel_list[l],channel_list[l+1], kernel_size = 3, padding = 1))
            layer_list.append(nn.ReLU())
        
        layer_list.append(nn.Flatten())
        
        self.Net = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        output = self.Net(x)
        return output
    
    
    
class Actor1_Arm(nn.Module):
    def __init__(self, shared_output_dim, arm_list, action1_dim):
        
        super(Actor1_Arm, self).__init__()
        
        self.input_dim = shared_output_dim
        self.arm_list = arm_list
        
        layer_list = []
        
        for l in range(len(arm_list)):
            if l == 0:
                layer_list.append(nn.Linear(self.input_dim, self.arm_list[l]))
                layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                layer_list.append(nn.Linear(self.arm_list[l-1], self.arm_list[l]))
                layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            
        layer_list.append(nn.Linear(self.arm_list[-1], action1_dim))
        layer_list.append(nn.Softmax(dim=-1))
        
        self.Act1 = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        output = self.Act1(x)
        return output
    
    
class Actor2_Arm(nn.Module):
    def __init__(self, shared_output_dim, arm_list, action2_dim):
        
        super(Actor2_Arm, self).__init__()
        
        self.input_dim = shared_output_dim
        self.arm_list = arm_list
        self.action2_dim = action2_dim
        
        layer_list = []
        
        for l in range(len(arm_list)):
            if l == 0:
                layer_list.append(nn.Linear(self.input_dim, self.arm_list[l]))
                #layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.Tanh())
                #layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                layer_list.append(nn.Linear(self.arm_list[l-1], self.arm_list[l]))
                #layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.Tanh())
                #layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            
        layer_list.append(nn.Linear(self.arm_list[-1], action2_dim))
        #layer_list.append(nn.Softmax(dim=-1))
        
        self.Act2 = nn.Sequential(*layer_list)
        
        
    def forward(self, x, mask):
        
        output = self.Act2(x)
        #print(output)
        output = torch.mul(torch.exp(output), mask) / torch.sum(torch.mul(torch.exp(output), mask))
        output = torch.nan_to_num(output, nan=1/3901, posinf=1/3901, neginf=1/3901)
        #print(output)
        #print(torch.sum(torch.mul(output,mask)))
        #output = torch.mul(output, mask)
        
        return output
    
    
class Critic_Arm(nn.Module):
    def __init__(self, shared_output_dim, arm_list):
        
        super(Critic_Arm, self).__init__()
        
        self.input_dim = shared_output_dim
        self.arm_list = arm_list
        
        layer_list = []
        
        for l in range(len(arm_list)):
            if l == 0:
                layer_list.append(nn.Linear(self.input_dim, self.arm_list[l]))
                #layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.Tanh())
                #layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                layer_list.append(nn.Linear(self.arm_list[l-1], self.arm_list[l]))
                #layer_list.append(nn.BatchNorm1d(self.arm_list[l]))
                layer_list.append(nn.Tanh())
                #layer_list.append(nn.LeakyReLU(negative_slope=0.2))
            
        layer_list.append(nn.Linear(self.arm_list[-1], 1))
        layer_list.append(nn.Tanh())
        #layer_list.append(nn.Sigmoid())
        
        self.Critic = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        output = self.Critic(x)
        #print(output)
        return output