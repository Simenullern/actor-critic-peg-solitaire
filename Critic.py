import numpy as np

class Critic:
    def __init__(self, use_nn=False, learning_rate=0.1,
                 elig_decay_rate=0.9, discount_factor=0.9):

        self.use_nn = use_nn
        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.value_func = dict()
        self.eligs = dict()

    def get_value(self, state):
        if state not in self.value_func.keys():
            self.value_func[state] = np.random.uniform(low=0, high=0.1)
        return self.value_func[state]

    def update_value_func(self, state, td_error):
        self.value_func[state] += \
            self.learning_rate * td_error * self.get_elig(state)
        return self.value_func[state]

    def get_elig(self, state):
        return self.eligs[state] if state in self.eligs.keys() \
            else 0

    def set_elig(self, state, value):
        self.eligs[state] = value

    def update_elig(self, state):
        if state not in self.eligs.keys():
            self.set_elig(state, 0)
        self.eligs[state] = self.discount_factor * self.elig_decay_rate * self.eligs[state]
        return self.eligs[state]

    def compute_TD_error(self, reward, state_value, succ_state_value):
        return reward + self.discount_factor * succ_state_value - state_value







