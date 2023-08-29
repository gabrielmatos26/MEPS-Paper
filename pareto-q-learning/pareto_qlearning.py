###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################


import numpy as np
import pygmo as pg
from pymoo.factory import get_performance_indicator
import random
from tqdm import tqdm, trange

class PQL:
    def __init__(self, hv_reference_point, hv_mapping_function):
        self.q_set = {}
        self.n_counter = {}
        self.average_reward = {}
        self.nd_policies = {}
        self.hv_calculator = get_performance_indicator("hv", ref_point=np.array(hv_reference_point))
        self.hv_mapping_function = hv_mapping_function

    def reset(self):
        self.q_set = {}
        self.n_counter = {}

    def hv_select_action(self, state, actions):
        hvs = []
        for action in actions:
            if (state,action) in self.q_set and len(self.q_set[(state, action)]) > 0:
                list_q_set = list(map(lambda x : list(x), self.q_set[(state,action)]))
                hv_points = np.array(list(map(self.hv_mapping_function, list_q_set)))
                hvs.append(self.hv_calculator.do(hv_points))
            else:
                hvs.append(0)
        
        hvs = np.array(hvs)
        if all(hvs == 0):
            return random.choice(actions)
        

        probs = hvs/hvs.sum()
        return int(np.random.choice(actions, p=probs))
        
        # return actions[np.argmax(hvs)]
    
    def cardinality_select_action(self, state, actions):
        cardinality = []
        for action in actions:
            if (state,action) in self.q_set and len(self.q_set[(state, action)]) > 0:
                list_q_set = np.array(list(map(lambda x: list(x), self.q_set[(state, action)])))
                if len(list_q_set) > 1:
                    ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*list_q_set)
                    cardinality.append(len(list_q_set[ndf[0]]))
                else:
                    cardinality.append(1)
            else:
                cardinality.append(0)
        
        cardinality = np.array(cardinality)
        if all(cardinality == 0):
            return random.choice(actions)

        probs = cardinality/cardinality.sum()
        return int(np.random.choice(actions, p=probs))

        # return actions[np.argmax(cardinality)]

    def pareto_select_action(self, state, actions):
        values = None
        for action in actions:
            if (state,action) in self.q_set and len(self.q_set[(state, action)]) > 0:
                list_q_set = np.array(list(map(lambda x: list(x), self.q_set[(state, action)])))
                action_array = np.ones((len(list_q_set),1)) * action
                list_q_set = np.concatenate((action_array, list_q_set), axis=1)
                if values is None:
                    values = list_q_set
                else:
                    values = np.concatenate((values, list_q_set), axis=0)
        
        if values is not None and len(values) > 0:
            if len(values) > 1:
                ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*values[:, 1:])

                non_dominated_actions, counts = np.unique(values[ndf[0], 0], return_counts=True)
                probs = counts/counts.sum()
                return int(np.random.choice(non_dominated_actions, p=probs))
            else:
                return int(values[0,0])
        else:
            return int(random.choice(actions))
    

    def select_action(self, state, actions, action_policy):
        if action_policy == 'hv':
            return self.hv_select_action(state, actions)
        elif action_policy == 'cardinality':
            return self.cardinality_select_action(state, actions)
        elif action_policy == 'pareto':
            return self.pareto_select_action(state, actions)
            pass 

    def set_vector_sum(self, v1, s1, gamma=1):
        array_v1 = np.array(list(v1))
        sum_set = set()
        for s1_element in map(lambda x: np.array(list(x)), s1):
            sum_set.add(tuple(gamma * s1_element + array_v1))
        
        return sum_set

    def update_nd_policies(self, state, action, next_state, possible_actions):
        if (state, action) not in self.nd_policies:
            self.nd_policies[(state, action)] = set()
        
        nd_sa = self.nd_policies[(state, action)]
        union_next_state_q_sets = set()
        for a in possible_actions:
            if (next_state, a) not in self.q_set:
                self.q_set[(next_state, a)] = set()

            q_next_sa = self.q_set[(next_state, a)]
            union_next_state_q_sets = union_next_state_q_sets.union(q_next_sa)
        
        if len(union_next_state_q_sets) > 0:
            list_union_next_state_q_sets = np.array(list(map(lambda x: list(x), union_next_state_q_sets)))
            if len(list_union_next_state_q_sets) > 1:
                ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*list_union_next_state_q_sets)
                nd_sa = set(map(lambda x: tuple(x), list_union_next_state_q_sets[ndf[0]]))
            else:
                nd_sa = set(map(lambda x: tuple(x), list_union_next_state_q_sets))
            self.nd_policies[(state, action)] = nd_sa

    def update_average_reward(self, state, action, reward):
        if (state, action) in self.average_reward:
            current_average_reward = self.average_reward[(state, action)]
            increment = (reward - current_average_reward) / self.n_counter[(state, action)]
            self.average_reward[(state, action)] += np.round(increment,2)
        else:
            self.average_reward[(state, action)] = reward.astype(float)

    def update_q_set(self, state, action, gamma):
        assert gamma >= 0 and gamma <= 1
        average_r = self.average_reward[(state, action)]
        nd_set = self.nd_policies[(state, action)]
        if len(nd_set) > 0:
            self.q_set[(state, action)] = self.set_vector_sum(average_r, nd_set, gamma)
        else:
            self.q_set[(state, action)] = set([tuple(average_r)])

    def epsilon_greedy_policy(self, selected_action, action_space, epsilon):
        action_probs = np.ones(len(action_space)) * epsilon / (len(action_space)-1)
        action_probs[selected_action] = (1.0 - epsilon)
        return np.random.choice(action_space, p=action_probs)

    def train(self, num_episodes, action_policy, env, gamma, num_steps=None, eps=0.997, n_iteration=1):
        # pbar_t = tqdm(total=num_episodes, leave=False)
        # pbar_t.set_postfix(PO = 0, HV = 0)
        # hv_list = np.zeros(num_episodes)
        for t in range(num_episodes):
            # current_eps = eps**(t+1 + n_iteration)
            current_eps = eps**n_iteration
            # pbar_t.set_description('EP {0}'.format(t+1))
            s = env.reset()
            is_terminal = env.is_terminal()
            i = 0
            while True:
                selected_action = self.select_action(s, env.get_action_space(), action_policy)
                selected_action = self.epsilon_greedy_policy(selected_action, env.get_action_space(), current_eps)
                s_next, rt, is_terminal = env.step(selected_action)
                if (s, selected_action) in self.n_counter:
                    self.n_counter[(s, selected_action)] += 1
                else:
                    self.n_counter[(s, selected_action)] = 1

                self.update_nd_policies(s, selected_action, s_next, env.get_action_space())
                self.update_average_reward(s, selected_action, rt)
                self.update_q_set(s, selected_action, gamma)

                s = s_next

                i += 1
                if is_terminal:
                    break
                elif num_steps is not None and i >= num_steps:
                    break
        
            # solutions, hv, num_solutions = self.test(gamma, env)
            # hv_list[t] = hv
            # pbar_t.set_postfix(PO = num_solutions, HV = hv)
            # pbar_t.update()
            
        # pbar_t.close()
        # return hv_list

    def compare_tuples(self, t1, t2, epsilon=1e-2):
        for i in range(len(t1)):
            if np.abs(t1[i] - t2[i]) > epsilon:
                return False
        return True

    def track_policy(self, target, gamma, env):
        s = env.reset()
        accumulated_reward = np.zeros(env.get_num_objectives())
        is_terminal = False
        if target is None:
            return accumulated_reward

        while not is_terminal:
            restart = False
            for action in env.get_action_space():
                if target is None:
                    return accumulated_reward
                if (s,action) in self.average_reward:
                    average_reward = self.average_reward[(s,action)]
                    nd = self.nd_policies.get((s,action), set())
                    if len(nd) > 0:
                        for q_vector in nd:
                            candidate_target = gamma * np.array(list(q_vector)) + np.array(list(average_reward)) 
                            if self.compare_tuples(tuple(candidate_target), tuple(target)):
                                # accumulated_reward += np.array(list(average_reward))
                                s, rt, is_terminal = env.step(action)
                                if is_terminal:
                                    accumulated_reward = rt
                                target = q_vector
                                restart = True
                                break
                    else:
                        candidate_target = np.array(list(average_reward)) 
                        if self.compare_tuples(tuple(candidate_target), tuple(target)):
                            # accumulated_reward += np.array(list(average_reward))
                            s, rt, is_terminal = env.step(action)
                            if is_terminal:
                                accumulated_reward = rt
                            target = None
                            restart = True
                if restart:
                    break
            else:
                accumulated_reward = rt
                break
        return accumulated_reward
    
    def test(self, gamma, env):
        initial_state = env.reset()
        union_nd_set = set()
        for action in env.get_action_space():
            union_nd_set = union_nd_set.union(self.q_set.get((initial_state, action), set()))

        union_nd_set = np.array(list(map(lambda x : list(x), union_nd_set)))
        if len(union_nd_set) > 1:
            ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*union_nd_set)
        else:
            return None, 0, 0
        union_nd_set = union_nd_set[ndf[0]]
        average_rewards = None

        for state_action_pair in union_nd_set:
            accumulated_reward = self.track_policy(tuple(state_action_pair), gamma, env)
            if average_rewards is None:
                average_rewards = accumulated_reward.reshape(1, -1)
            else:
                average_rewards = np.concatenate((average_rewards, accumulated_reward.reshape(1, -1)))

            
        # average_rewards = np.array(average_rewards)
        if len(average_rewards) > 1:
            ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*average_rewards)
            average_rewards = average_rewards[ndf[0]]

        hv_points = np.array(list(map(self.hv_mapping_function, average_rewards)))
        hv = self.hv_calculator.do(hv_points)
        num_solutions = len(average_rewards)
        # print("HV {0}".format(hv))
        # print("Num solutions {0}".format(num_solutions))
        # print("Solutions:")
        # print(average_rewards)
        return average_rewards, hv, num_solutions