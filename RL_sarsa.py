from RL_Environment import Env
from collections import defaultdict
import numpy as np
import random

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.discount_factor = 0.9
        self.learning_rate = 0.01
        self.epsilon = 0.1
        # For the Q_table, initialize possible actions
        self.q_table = defaultdict(lambda : [0.0], [0.0], [0.0], [0.0])

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # return random action
            action = np.random.choice(self.action)
        else:
            # Retrun accoding to Q-func
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        return action

    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor
                                                   * next_state_q - current_q))
        self.q_table[state][action] = new_q

    @staticmethod
    def arg_max(self, state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)

            return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = SARSAgent()

