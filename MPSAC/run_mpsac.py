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

from errno import EEXIST
from warnings import resetwarnings
from sac_v2 import *
import numpy
import argparse

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
from deap import cma
from deap import creator
from deap import tools

import os

from pymoo.indicators.hv import HV
from tqdm import tqdm, trange
from datetime import datetime
import pickle
import torch

from parallel import ParallelEvaluator

def evaluate_net(env_class, sac_trainer, max_ep_len, individual, explore=False, return_action_list=False, residential=None):
        if residential is None:
            env = env_class()
        else:
            env = env_class(residential)
        state = env.reset()
        action_list = []
        buffer_list = []
        for step in range(max_ep_len):
            if not explore:
                action = sac_trainer.policy_net.get_action(state)
            else:
                action = np.random.randint(len(env.get_action_space()))
            action_list.append(action)
            next_state, reward, done = env.step(action)
            buffer_list.append((state, action, reward, next_state, done, individual))
            
            state = next_state
            
            if done:
                break
        if return_action_list:
            return reward, buffer_list, action_list
        return reward, buffer_list

def evaluate_cma_net(env_class, individual, max_ep_len, num_hidden=1, hidden_dim = 128, state_dim=1, residential=None):
    if residential is None:
        env = env_class()
    else:
        env = env_class(residential)
    action_dim = len(env.get_action_space())
    individual_net = SAC_Individual(0, state_dim, action_dim, None, hidden_dim, 1, None, num_hidden)
    individual_net.unflatten(individual)
    state = env.reset()
    for step in range(max_ep_len):
        action = individual_net.policy_net.get_action(state)
        
        next_state, reward, done = env.step(action)
        
        state = next_state
        
        if done:
            break
    del individual_net
    return reward

class MPSAC:
    def __init__(self):
        self.hv = None

    def load_env(self, name):
        if name == "dst":
            from envs.DST.DeepSeaTreasureEnv import DeepSeaTreasureEnv
            self.hv = HV(ref_point=np.array([25, 0]))
            self.env = DeepSeaTreasureEnv
        else:
            raise Exception("Inexistent environment")
        self.model_path = './envs/{}/model'.format(name.upper())
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

    def evaluate_net(self, sac_trainer, max_ep_len, individual, explore=False, return_action_list=False, residential=None):
        if residential is None:
            env = self.env(False)
        else:
            env = self.env(residential)
        state = env.reset()
        action_list = []
        buffer_list = []
        for step in range(max_ep_len):
            if not explore:
                action = sac_trainer.policy_net.get_action(state)
            else:
                action = np.random.randint(len(env.get_action_space()))
            action_list.append(action)
            next_state, reward, done = env.step(action)
            buffer_list.append((state, action, reward, next_state, done, individual))
            
            state = next_state
            
            if done:
                break
        if return_action_list:
            return reward, buffer_list, action_list
        return reward, buffer_list

    def evaluate_cma_net(self, individual, max_ep_len, num_hidden=1, hidden_dim = 128, state_dim=1, residential=None):
        if residential is None:
            env = self.env(False)
        else:
            env = self.env(residential)
        action_dim = len(env.get_action_space())
        individual_net = SAC_Individual(0, state_dim, action_dim, None, hidden_dim, 1, None, num_hidden)
        individual_net.unflatten(individual)
        state = env.reset()
        for step in range(max_ep_len):
            action = individual_net.policy_net.get_action(state)
           
            next_state, reward, done = env.step(action)
            
            state = next_state
            
            if done:
                break
        del individual_net
        return reward
        

    def run_mo_cma_es(self, ngen, population_net, max_ep_len, num_hidden=1, hidden_dim = 128, state_dim=1, residential=None):
        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.evaluate_cma_net)

        if residential is None:
            env = self.env(False)
        else:
            env = self.env(residential)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,) * env.get_num_objectives())
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        MU = len(population_net)
        LAMBDA = len(population_net)

        # The MO-CMA-ES algorithm takes a full population as argument
        population = [creator.Individual(x.flatten()) for x in population_net]

        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind, max_ep_len, num_hidden, hidden_dim, state_dim, residential)
        
        strategy = cma.StrategyMultiObjective(population, sigma=1.0, mu=MU, lambda_=LAMBDA)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hv_history = []

        with trange(ngen) as t:
            for i in t:
                t.set_description('GEN {0}'.format(i+1))
                fitness_history = []
                # Generate a new population
                population = toolbox.generate()

                for ind in population:
                    ind.fitness.values = toolbox.evaluate(ind, max_ep_len, num_hidden, hidden_dim, state_dim, residential)

                toolbox.update(population)

                for ind in strategy.parents:
                    fitness_history.append(ind.fitness.values)
                current_hv = self.hv.do(-1*np.array(fitness_history))
                hv_history.append(current_hv)

                t.set_postfix(HV = str(current_hv))

        return hv_history, numpy.asarray(strategy.parents)
    
    def run(self, env_name, popsize=10, num_hidden=1, hidden_dim = 128, total_runs = 1, max_ep_len=40,\
         sac_episodes = 1000, mo_cma_es_episodes = 1000, AUTO_ENTROPY=True, DETERMINISTIC=True, residential=None):
        results = {}
        for r in range(total_runs):
            print("Run {}".format(r+1))
            self.load_env(env_name)
            if residential is None:
                env = self.env(False)
            else:
                env = self.env(residential)
            num_objectives = env.get_num_objectives()
            replay_buffer_size = 1e5 #1e6
            self.replay_buffer = MultiChannelReplayBuffer(replay_buffer_size, popsize)
            action_dim = len(env.get_action_space())
            state_dim  = 3
            action_range = 1 #max(self.env.get_action_space())
        
            update_itr = 1
            hv_hist = []
            current_hv = 0
            batch_size = 32
            sampling_iteractions = 25 #50 #100
            training_iteractions = 25 #50 #100
            samples_from_each = 25
            #define a mini batch 
            sac_trainer=SAC_Trainer(state_dim, action_dim, self.replay_buffer, hidden_dim=hidden_dim, num_hidden=num_hidden, action_range=action_range, num_objectives=num_objectives, num_networks=popsize)

            with trange(sac_episodes) as t:
                for i in t:
                    t.set_description('GEN {0}'.format(i+1))
                    for it in trange(sampling_iteractions + training_iteractions, leave=False):
                        rewards = []
                        for ind in range(popsize):
                            reward, buffer_list = self.evaluate_net(sac_trainer.individuals[ind], max_ep_len, ind, explore = it < sampling_iteractions, residential=residential)
                            self.replay_buffer.push_list(buffer_list)
                            rewards.append(reward)

                        if it > sampling_iteractions:
                            for _ in range(update_itr):
                                sac_trainer.update(batch_size, auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim, samples_from_each=samples_from_each)
                    
                        current_hv = self.hv.do(-1*np.array(rewards))
                        hv_hist.append(current_hv)
                        t.set_postfix(HV = str(current_hv))

                    if i % 20 == 0 and i > 0: # plot and model saving interval
                        np.save(os.path.join(self.model_path, 'hvs'), rewards)
            
            memory_best_list = [self.evaluate_net(sac_trainer.individuals[ind], max_ep_len, ind, explore = False, residential=residential)[0] for ind in range(popsize)]
            memory_best_list.sort(key=lambda x: x[0])
            for solution in memory_best_list:
                print("$/kWh {0}, REF {1}, Degradation {2}".format(*np.abs(solution)))

            hv_hist_cma, new_individuals = self.run_mo_cma_es(mo_cma_es_episodes, sac_trainer.individuals, max_ep_len, num_hidden, hidden_dim, state_dim, residential)
            hv_hist += hv_hist_cma
            sac_trainer.unflatten(new_individuals)
            final_rewards = []
            final_policies = []
            for ind in range(popsize):
                reward, _, action_list = self.evaluate_net(sac_trainer.individuals[ind], max_ep_len, ind, explore = False,\
                     return_action_list=True, residential=residential)
                final_rewards.append(np.abs(reward))
                final_policies.append(action_list)

            results[r+1] = {'hv':hv_hist, 'front': np.array(final_rewards), 'policies':final_policies}
            for i, solution in enumerate(sorted(final_rewards, key=lambda x : x[0])):
                print("$/kWh {0}, REF {1}, Degradation {2}".format(*np.abs(solution)))

            with open(os.path.join(self.model_path, 'results_hvc_{0}_{1}.pickle'.format(r+1, datetime.now().strftime("%d-%m-%Y_%H-%M"))), 'wb') as f:
                pickle.dump(results, f)
        with open(os.path.join(self.model_path, 'results_hvc_{0}.pickle'.format(datetime.now().strftime("%d-%m-%Y_%H-%M"))), 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MPSAC')
    parser.add_argument('--env', dest='env', action='store', default="dst")
    parser.add_argument('--popsize', dest='popsize', action='store', type=int, default=50)
    parser.add_argument('--residential', dest='residential', action='store', type=int, default=None)

    args = parser.parse_args()
    residential = args.residential > 0 if args.residential is not None else None
    alg = MPSAC()
    alg.run(args.env, popsize=args.popsize, num_hidden=1, hidden_dim = 64, total_runs = 20,\
         max_ep_len=12, sac_episodes = 5, mo_cma_es_episodes = 250, AUTO_ENTROPY=False, DETERMINISTIC=True, residential=residential)
  
