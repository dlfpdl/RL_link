from RL_Environment import GraphicDisplay, Env

class Value_Iteration:
    def __init__(self):
        self.env = env
        self.value_table = [[0.00] * env.width for _ in range(self.env.height)]
        self.discount_factor = 0.9

    def value_iteration(self):
        next_value_table = [[0.00] * self.env.height for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue

            # Empty list for value_function
            value_list = []

            # Calculating all possible actions
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))

            # 최댓값을 다음 가치함수로 대입
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # Calculating that from all function to
        # Q-func = (r + (discout_rate * next_state_value_func))

        for action in self.env.possible_actions:
            next_state = self.env.get_reward(state, action)
            reward = self.env.get_reward(next_state)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                value = max_value.append(value)
            elif value == max_value:
                action_list.append(action)

            return action_list

if __name__ == "__main__":
    env = Env()
    value_iteration = Value_Iteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
