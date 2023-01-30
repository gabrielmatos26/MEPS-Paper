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

import numpy as np
import math

class DeepSeaTreasureEnv:
    def __init__(self):
        ## specify the depths and treasures associated with each depth
        self.map = np.array([
            [ 0,  0,   0,  0,  0,  0,  0,  0,   0,   0],
            [ 1,  0,   0,  0,  0,  0,  0,  0,   0,   0],
            [-1,  2,   0,  0,  0,  0,  0,  0,   0,   0],
            [-1, -1,   3,  0,  0,  0,  0,  0,   0,   0],
            [-1, -1,  -1,  5,  8, 16,  0,  0,   0,   0],
            [-1, -1,  -1, -1, -1, -1,  0,  0,   0,   0],
            [-1, -1,  -1, -1, -1, -1,  0,  0,   0,   0],
            [-1, -1,  -1, -1, -1, -1, 24, 50,   0,   0],
            [-1, -1,  -1, -1, -1, -1, -1, -1,   0,   0],
            [-1, -1,  -1, -1, -1, -1, -1, -1,  74,   0],
            [-1, -1,  -1, -1, -1, -1, -1, -1,  -1, 124]
        ])

        ## define map size
        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]

        ## initializes agent in the top left corner
        self.agent_row = 0
        self.agent_col = 0

        self.current_treasure = 0
        # self.time = 0
        self.time = 0

    
    def is_valid(self, row, col):
        if col >= self.num_cols or col < 0:
            return False
        elif row >= self.num_rows or row < 0:
            return False
        elif self.map[row, col] == -1:
            return False
        return True

    def update_position(self, action):
        new_row = self.agent_row
        new_col = self.agent_col
        ## go right
        if action == 0:
            ## increase column
            tmp_new_col = self.agent_col + 1
            if self.is_valid(self.agent_row, tmp_new_col):
                new_col = tmp_new_col
        ## go left
        elif action == 1:
            ## decrease column
            tmp_new_col = self.agent_col - 1
            if self.is_valid(self.agent_row, tmp_new_col):
                new_col = tmp_new_col
        ## go down
        elif action == 2:
            ## increase row
            tmp_new_row = self.agent_row + 1
            if self.is_valid(tmp_new_row, self.agent_col):
                new_row = tmp_new_row
        ## go up
        elif action == 3:
            ## decrease row
            tmp_new_row = self.agent_row - 1
            if self.is_valid(tmp_new_row, self.agent_col):
                new_row = tmp_new_row

        return new_row, new_col

    def get_reward(self, action=None):
        if action is not None:
            new_row, new_col = self.update_position(action)
            treasure = self.map[new_row,new_col]
            if treasure > 0:
                return np.array([self.time - 1, treasure])
            return np.array([self.time - 1, 0])
        else:
            return np.array([self.time, self.current_treasure])

    def get_reward_normalized(self, max_time):
        max_treasure = np.max(self.map)
        norm_const = np.array([max_time, max_treasure])
        return self.get_reward()/norm_const

    def is_terminal(self):
        if self.current_treasure != 0:
            return True
        return False

    def step(self, action):
        self.time, self.current_treasure = self.get_reward(action)
        self.agent_row, self.agent_col = self.update_position(action)

        return self.get_state(), self.get_reward(), self.is_terminal()
    
    def get_action_space(self):
        return [0,1,2,3]

    def get_num_objectives(self):
        return 2
    
    def get_state(self):
        state = self.agent_row * self.num_cols + self.agent_col
        return state + 1 # avoid getting 0 state
        # return self.agent_col * self.num_rows + self.agent_row
    
    def reset(self):
        ## reinitializes agent in the top left corner
        self.agent_row = 0
        self.agent_col = 0

        self.current_treasure = 0
        self.time = 0


        # self.agent_row = 9
        # self.agent_col = 9

        # self.current_treasure = 0
        # self.time = 0

        return self.get_state()

