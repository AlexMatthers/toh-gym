#!/usr/bin/env python
# coding: utf-8

# # Tower of Hanoi
# Taken and modified from https://github.com/xadahiya/toh-gym
import sys
from contextlib import closing

import numpy as np
from six import StringIO, b
from gym.envs.toy_text import discrete
import random
from itertools import permutations as perm

class TohEnv(discrete.DiscreteEnv):
    """Tower of hanoi environment."""
    metadata = {'render.modes': ['human', 'ansi']}

    def apply_action(self, s, a):
        """Apply a move and generate the new state, if move is invalid, return None."""
        s = [list(i) for i in s]
        if len(s[a[0]]) == 0:
            # Invalid move
            return None
        to_move = s[a[0]][-1]
        source = a[0]
        dest = a[1]
        new_state = s[:]
        new_state[source].pop()
        new_state[dest].append(to_move)
        output = tuple([tuple(i) for i in new_state])
        return output

    def is_state_valid(self, s):
        """Checks if a state is valid."""
        s = [list(i) for i in s]
        for i in s:
            if i != sorted(i, reverse=True):
                return False
        return True

    def generate_all_states(self):
        """Generate all the states for MDP, total number of states = number_of_poles**number_of_disks"""
        states = []
        states.append(self.initial_state)

        while True:
            old_len = len(states)
            for s in states:
                for action in self.action_list:
                    new_state = self.apply_action(s, action)
                    if new_state and new_state not in states:
                        if self.is_state_valid(new_state):
                            states.append(new_state)
            new_len = len(states)
            if old_len == new_len:
                break
        return states

    def randomValid(self, state):
        pass

    def __init__(self, poles=3, rings=3, noise=0, stepReward=0, invalidReward=0):

        self.initial_state = tuple([tuple(range(rings, 0, -1))] + [()] * (poles - 1))
        assert noise < 1.0, "noise must be between 0 and 1"
        self.goal_state = tuple([()] * (poles - 1) + [tuple(range(rings, 0, -1))])

        self.action_list = np.array(list(perm(range(poles), 2)))

        self.all_states = self.generate_all_states()

        self.nS = len(self.all_states)
        self.nA = len(self.action_list)

        # Maintaining mappings to make use of algorithms from frozen lake.
        # Used to get a state by index of an array instead of a tuple
        self.state_mapping = {}
        self.inverse_mapping = {}
        for i in range(len(self.all_states)):
            self.state_mapping[i] = self.all_states[i]
            self.inverse_mapping[self.all_states[i]] = i

        ## Generating probability matrix
        self.P = {s: {a: [] for a in range(len(self.action_list))}
                  for s in range(len(self.all_states))}

        # For stochastic environment
        goalR = 1
        if stepReward:
            stepR = stepReward
        else:
            stepR = 0
        if invalidReward:
            invalidR = invalidReward
        else:
            invalidR = 0
        self.noise = noise
        for s in range(len(self.all_states)):
            for a in range(len(self.action_list)):
                li = self.P[s][a]
                if self.state_mapping[s] == self.goal_state:
                    li.append((1, s, 0, True))
                else:
                    if noise == 0:
                        done = False
                        new_state = self.apply_action(
                            self.state_mapping[s], self.action_list[a])
                        rew = stepR
                        if new_state == None:
                            new_state = self.state_mapping[s]
                        if self.is_state_valid(new_state) == False:
                            new_state = self.state_mapping[s]
                            rew = invalidR
                            done = True
                        if new_state == self.goal_state:
                            rew = goalR
                            done = True
                        li.append(
                            (1, self.inverse_mapping[new_state], rew, done))
                    else:
                        for b in [(a, 1-noise), (np.random.choice(range(6)), noise)]:
                            a, prob = b[0], b[1]
                            done = False
                            new_state = self.apply_action(
                                self.state_mapping[s], self.action_list[a])
                            rew = stepR
                            if new_state == None:
                                new_state = self.state_mapping[s]
                            if self.is_state_valid(new_state) == False:
                                new_state = self.state_mapping[s]
                                rew = invalidR
                                done = True
                            if new_state == self.goal_state:
                                rew = goalR
                                done = True
                            li.append(
                                (prob, self.inverse_mapping[new_state], rew, done))

        self.isd = np.array([self.is_state_valid(self.state_mapping[s])
                             for s in range(len(self.all_states))]).astype('float').ravel()
        self.isd /= self.isd.sum()

        super(TohEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        currState = [list(s) for s in self.state_mapping[self.s]]
        ringN = np.max(self.initial_state)[0]
        poleN = len(self.initial_state)
        pole = ' ' * ringN + '||' + ' ' * ringN
        rings = []
        for r in range(1, ringN + 1):
            rings.append(' ' * (ringN - r) + '~' * r + '||' + '~' * r + ' ' * (ringN - r))

        rings = np.array(rings)
        floor = '\u203e' * (ringN * 2 + 4)
        vis = []
        for p in currState:
            p = np.array(p) - 1
            if p.size == 0:
                vis.append([pole] * (ringN + 1))
            else:
                temp = rings[p[::-1]]
                vis.append([pole] * (ringN - temp.size + 1) + temp.tolist())

        vis = ''.join([''.join(i) + '\n' for i in np.array(vis).T.tolist()]) + floor * poleN + '\n'

        if self.lastaction is not None:
            outfile.write('Pole {} to Pole {}\n'.format(*(self.action_list[self.lastaction]) + 1))
        else:
            outfile.write('\n')
        outfile.write(vis)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
