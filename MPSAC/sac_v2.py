###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################
# Copyright (c) 2023, Gabriel Matos Leite
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader, TensorDataset

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from joblib import Parallel, delayed

import argparse
import time

GPU = False #True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class MultiChannelReplayBuffer:
    def __init__(self, capacity, channels=1):
        self.capacity = capacity
        self.buffers = dict([(i, ReplayBuffer(capacity)) for i in range(channels)])
    
    def push(self, state, action, reward, next_state, done, channel):
        channel_buffer = self.buffers[channel]
        if len(channel_buffer.buffer) < self.capacity:
            channel_buffer.buffer.append(None)
        channel_buffer.buffer[channel_buffer.position] = (state, action, reward, next_state, done)
        channel_buffer.position = int((channel_buffer.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, weight, samples_from_each=1):
        batch = []
        for _, channel_buffer in self.buffers.items():
            batch.append(channel_buffer.sample(samples_from_each))
        buffer_tuple = map(np.stack, zip(*batch))
        buffer_list = []
        for element in buffer_tuple:
            if len(element.shape) > 2:
                buffer_list.append(element.reshape(np.array(element.shape[:-1]).prod(), element.shape[-1]).squeeze())
            else:
                buffer_list.append(element.reshape(np.array(element.shape).prod(),1))

        state, action, reward, next_state, done = buffer_list
        return state, action, weight @ reward.T, next_state, done
    
    def __len__(self):
        return len(self.buffers)
    

        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_hidden=1, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        assert num_hidden > 0
        
        layers = [nn.Linear(num_inputs, hidden_dim), nn.ReLU()]
        if num_hidden > 1:
            for _ in range(num_hidden-1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_actions))
        
        # weights initialization
        layers[-1].weight.data.uniform_(-init_w, init_w)
        layers[-1].bias.data.uniform_(-init_w, init_w)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, state, action=None):
        if action is None:
            return self.layers(state)
        return self.layers(state).gather(-1, action.long())
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_hidden=1, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        assert num_hidden > 0
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = [nn.Linear(num_inputs, hidden_dim), nn.ReLU()]
        if num_hidden > 1:
            for _ in range(num_hidden-1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

        layers += [nn.Linear(hidden_dim, num_actions), nn.Softmax(dim=-1)]
        self.layers = nn.Sequential(*layers)
        
        self.num_actions = num_actions

        
    def forward(self, state, action=None):
        if action is None:
            return self.layers(state)
        return self.layers(state).gather(-1, action.long())
    
    def evaluate(self, state, epsilon=1e-8, action=None):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.forward(state, action)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)
        new_action = probs.argmax(dim=1).unsqueeze(1)
        return log_probs, new_action
        
    
    def get_action(self, state, deterministic=True):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor([state]).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action



class SAC_Trainer():
    def __init__(self, state_dim, action_dim, replay_buffer, hidden_dim, action_range, num_hidden=1, num_networks=10, num_objectives=2):
        self.replay_buffer = replay_buffer
        self.num_networks = num_networks
        self.weights = self.generate_random_weights(num_networks, num_objectives)
        self.individuals = [SAC_Individual(i, state_dim, action_dim, replay_buffer, hidden_dim, action_range, self.weights[i,:], num_hidden) for i in range(num_networks)]
        

    def simplex_lattice_design(self, H, num_objectives=2):
        assert num_objectives < 4 and num_objectives > 0
        assert H > 0
        num_rows = int(np.math.factorial(H + num_objectives - 1) / (np.math.factorial(num_objectives - 1) * np.math.factorial(H)))
        w = np.zeros((num_rows, num_objectives))
        if num_objectives == 2:
            w[:,0] = np.linspace(0, 1, H+1)
            w[:,1] = 1 - w[:,0]
        else:
            w[:,0] = np.repeat(np.linspace(0,1,H+1), np.arange(1, H+2)[::-1])
            w[:,1] = np.concatenate([np.linspace(0,1,H+1)[::-1][i:][::-1].tolist() for i in range(H+1)])
            w[:,2] = 1 - w[:,0] - w[:,1]
        idx = np.arange(len(w))
        random.shuffle(idx)
        return w[idx[:H], :]
        
    def generate_paper_weights(self, H, num_objectives=2):
        assert num_objectives < 4 and num_objectives > 0
        assert H > 0
        second_obj = np.linspace(0,1,H)
        first_obj = np.ones(H)
        w = np.stack((first_obj, second_obj), axis=1)
        predefined_w = np.array([[1.0, 0.1], [1.0, 0.12], [1.0, 0.14], [1.0, 0.16], [1.0, 0.18]])

        idx = np.arange(len(w))
        random.shuffle(idx)
        w = np.vstack((predefined_w, w[idx,:]))
        w = w / w.sum(axis=1).reshape(-1,1)
        return w[:H, :]

    def generate_random_weights(self, H, num_objectives=2):
        assert num_objectives < 4 and num_objectives > 0
        assert H > 0
        w = np.random.random((H, num_objectives))
        w = w / w.sum(axis=1).reshape(-1,1)
        return w

    
    def update(self, batch_size, auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2, samples_from_each=1):
        for individual in self.individuals:
            individual.update(batch_size, auto_entropy, target_entropy, gamma,soft_tau, samples_from_each)
    
    def unflatten(self, individuals_flattened):
        for i, individual in enumerate(self.individuals):
            individual.unflatten(individuals_flattened[i])

    def save_model(self, path):
        for individual in self.individuals:
            individual.save_model(path)

    def load_model(self, path):
        for individual in self.individuals:
            individual.load_model(path)
    


class SAC_Individual():
    def __init__(self, index, state_dim, action_dim, replay_buffer, hidden_dim, action_range, weight, num_hidden=1):
        self.index = index
        self.replay_buffer = replay_buffer
        self.weight = weight

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, num_hidden=num_hidden).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, num_hidden=num_hidden).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, num_hidden=num_hidden).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, num_hidden=num_hidden).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, num_hidden=num_hidden, action_range=action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 1e-3
        policy_lr = 1e-3
        alpha_lr  = 1e-3

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.soft_q_net1_dict = dict([(i, parameters.data.detach().cpu().numpy().shape) for i, parameters in enumerate(self.soft_q_net1.parameters())])
        self.soft_q_net2_dict = dict([(i, parameters.data.detach().cpu().numpy().shape) for i, parameters in enumerate(self.soft_q_net2.parameters())])
        self.target_soft_q_net1_dict = dict([(i, parameters.data.detach().cpu().numpy().shape) for i, parameters in enumerate(self.target_soft_q_net1.parameters())])
        self.target_soft_q_net2_dict = dict([(i, parameters.data.detach().cpu().numpy().shape) for i, parameters in enumerate(self.target_soft_q_net2.parameters())])
        self.policy_net_dict = dict([(i, parameters.data.detach().cpu().numpy().shape) for i, parameters in enumerate(self.policy_net.parameters())])

    
    def update(self, batch_size, auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2, samples_from_each=1):
        state_list, action_list, reward_list, next_state_list, done_list = self.replay_buffer.sample(self.weight, samples_from_each)
        
        state_tensor      = torch.FloatTensor(state_list).to(device)
        next_state_tensor = torch.FloatTensor(next_state_list).to(device)
        action_tensor     = torch.FloatTensor(action_list).to(device)
        reward_tensor     = torch.FloatTensor(reward_list.reshape(-1,1)).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done_tensor       = torch.FloatTensor(np.float32(done_list)).to(device)

        dataset = TensorDataset(state_tensor, next_state_tensor, action_tensor, reward_tensor, done_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for id_batch, (state, next_state, action, reward, done) in enumerate(dataloader):

            predicted_q_value1 = self.soft_q_net1(state, action)
            predicted_q_value2 = self.soft_q_net2(state, action)
            log_prob, _ = self.policy_net.evaluate(state, action)
            with torch.no_grad():
                next_log_prob, next_action = self.policy_net.evaluate(next_state)

            self.alpha = self.log_alpha.exp()

            target_q_min = (next_log_prob.exp() * \
                (torch.min(self.target_soft_q_net1(next_state, next_action),self.target_soft_q_net2(next_state, next_action)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
            target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            self.soft_q_optimizer2.step()  

        # Training Policy Function
            with torch.no_grad():
                predicted_new_q_value = torch.min(self.soft_q_net1(state, action),self.soft_q_net2(state, action))
            policy_loss = (log_prob.exp()*(self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Updating alpha wrt entropy
            # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
            if auto_entropy is True:
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            else:
                self.alpha = 1.
                alpha_loss = 0

            # Soft update the target value net
            for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )


    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_{}_q1'.format(self.index))
        torch.save(self.soft_q_net2.state_dict(), path+'_{}_q2'.format(self.index))
        torch.save(self.policy_net.state_dict(), path+'_{}_policy'.format(self.index))

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_{}_q1'.format(self.index)))
        self.soft_q_net2.load_state_dict(torch.load(path+'_{}_q2'.format(self.index)))
        self.policy_net.load_state_dict(torch.load(path+'_{}_policy'.format(self.index)))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
    
    def flatten(self):
        soft_q_net1_flattened = np.concatenate([parameters.data.detach().cpu().numpy().flatten().tolist() for parameters in self.soft_q_net1.parameters()])
        soft_q_net2_flattened = np.concatenate([parameters.data.detach().cpu().numpy().flatten().tolist() for parameters in self.soft_q_net2.parameters()])
        target_soft_q_net1_flattened = np.concatenate([parameters.data.detach().cpu().numpy().flatten().tolist() for parameters in self.target_soft_q_net1.parameters()])
        target_soft_q_net2_flattened = np.concatenate([parameters.data.detach().cpu().numpy().flatten().tolist() for parameters in self.target_soft_q_net2.parameters()])
        policy_net_flattened = np.concatenate([parameters.data.detach().cpu().numpy().flatten().tolist() for parameters in self.policy_net.parameters()])

        flattened_vec = np.concatenate((soft_q_net1_flattened,\
             soft_q_net2_flattened,\
             target_soft_q_net1_flattened,\
             target_soft_q_net2_flattened,\
             policy_net_flattened))
        return flattened_vec
    
    def unflatten(self, flattened_vec):
        self.soft_q_net1, pointer = self.unflatten_net(flattened_vec, self.soft_q_net1, 1, 0)
        self.soft_q_net2, pointer = self.unflatten_net(flattened_vec, self.soft_q_net2, 2, pointer)
        self.target_soft_q_net1, pointer = self.unflatten_net(flattened_vec, self.target_soft_q_net1, 3, pointer)
        self.target_soft_q_net2, pointer = self.unflatten_net(flattened_vec, self.target_soft_q_net2, 4, pointer)
        self.policy_net, pointer = self.unflatten_net(flattened_vec, self.policy_net, 5, pointer)
    
    def unflatten_net(self, flattened_vec, net, type, pointer):
        if type == 1: #soft_q_1
            net_dict = self.soft_q_net1_dict
        elif type == 2: #soft_q_2
            net_dict = self.soft_q_net2_dict
        elif type == 3: #target_q_1
            net_dict = self.target_soft_q_net1_dict
        elif type == 4: #target_q_2
            net_dict = self.target_soft_q_net2_dict
        elif type == 5: #policy_net
            net_dict = self.policy_net_dict
        
        prev_idx, next_idx = 0, 0
        for _, shape in net_dict.items():
            next_idx += np.array(shape).prod()
        prev_idx += pointer
        next_idx += pointer
        net_flattened = flattened_vec[prev_idx:next_idx]

        net_params = []
        prev_idx = 0
        for _, shape in net_dict.items():
            next_idx = prev_idx + np.array(shape).prod()
            net_params.append(np.array(net_flattened[prev_idx:next_idx]).reshape(shape))
            prev_idx = next_idx
        
        for target_param, param in zip(net.parameters(), net_params):
                target_param.data.copy_(torch.from_numpy(param))
        return net, next_idx



