from Funcapp import Funcapp
import torch
import numpy as np

class Critic:
    def __init__(self, layers, use_nn=False, learning_rate=0.1,
                 elig_decay_rate=0.9, discount_factor=0.9):

        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.value_func = dict()
        self.eligs = dict()
        self.use_nn = use_nn
        self.funcapp = None
        if use_nn:
            self.funcapp = Funcapp(layers=layers).double()

        #abc = torch.tensor(np.array([1.0 for i in range(0, 15)]), dtype=torch.float)
        #x = self.funcapp.forward(abc)
        #breakpoint()


    def init_state_value_if_needed(self, state):
        if state not in self.value_func.keys():
            self.value_func[state] = 0

    def get_value(self, state):
        self.init_state_value_if_needed(state)
        return self.value_func[state]

    def update_value_func(self, state, td_error, critic_elig):
        self.value_func[state] = self.value_func[state] + self.learning_rate * td_error * critic_elig
        return self.value_func[state]

    def get_elig(self, state):
        return self.eligs[state]

    def set_elig(self, state, value):
        self.eligs[state] = value

    def update_elig(self, state, critic_elig):
        self.eligs[state] = self.discount_factor * self.elig_decay_rate * critic_elig
        return self.eligs[state]

    def reset_eligs(self):
        for key in self.eligs.keys():
            self.eligs[key] = 0

    def compute_TD_error(self, reward, state_value, succ_state_value):
        return reward + self.discount_factor * succ_state_value - state_value
