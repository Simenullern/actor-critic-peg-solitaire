import torch
import numpy as np
from Funcapp import Funcapp

class Critic:
    def __init__(self, layers, use_nn, learning_rate, elig_decay_rate, discount_factor):

        self.use_nn = use_nn
        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.value_func = None if use_nn else dict()
        self.eligs = [] if use_nn else dict()
        self.funcapp = Funcapp(self, layers, learning_rate) if use_nn else None

    def init_state_value_if_needed(self, state):
        if not self.use_nn and state not in self.value_func.keys():
            self.value_func[state] = 0

    def get_value(self, state):
        if self.use_nn:
            X = self.vectorize_state(state)
            return self.funcapp.forward(X)
        else:
            self.init_state_value_if_needed(state)
            return self.value_func[state]

    def update_value_func(self, state, td_error, critic_elig):
        if self.use_nn:
            # STEP 1: backprop TD_error to get gradients of loss
            X = Critic.vectorize_state(state)
            y_pred = self.funcapp.forward(X)
            y_target = td_error + y_pred
            loss = self.funcapp.criterion(y_pred, y_target)
            loss.backward(retain_graph=True)

            # STEP 2: Update eligs in the direction of the partial derivative of value func w.rt. weights
            new_eligs = []
            for i in range(0, len(self.funcapp.net) - 1):
                deriv_val_func = (self.funcapp.net[i].weight.grad / -td_error).detach().numpy()
                new_eligs.append(np.add(self.eligs[i], deriv_val_func))
            self.eligs = np.array(new_eligs)

            # STEP 3: Now update gradients with the elig contribution
            for i in range(0, len(self.funcapp.net) - 1):
                self.funcapp.net[i].weight.grad *= td_error * torch.tensor(self.eligs[i], dtype=torch.float)

            # STEP 4: Now can the weights be updated
            self.funcapp.optimizer.step()
            self.funcapp.optimizer.zero_grad()
            return y_pred

        else:
            self.value_func[state] = self.value_func[state] + self.learning_rate * td_error * critic_elig
            return self.value_func[state]

    def get_elig(self, state):
        return self.eligs if self.use_nn else self.eligs[state]

    def set_elig(self, state, value):
        if not self.use_nn:
            self.eligs[state] = value

    def update_elig(self, state, critic_elig):
        if self.use_nn:
            for i in range(0, len(self.eligs)):
                self.eligs = self.discount_factor * self.elig_decay_rate * critic_elig
            return self.eligs
        else:
            self.eligs[state] = self.discount_factor * self.elig_decay_rate * critic_elig
            return self.eligs[state]

    def reset_eligs(self):
        if self.use_nn:
            self.set_elig(None, 0)
        else:
            for key in self.eligs.keys():
                self.eligs[key] = 0

    def compute_TD_error(self, reward, state_value, succ_state_value):
        return reward + self.discount_factor * succ_state_value - state_value

    @staticmethod
    def vectorize_state(state):
        out = []
        for char in state:
            if char == '1':
                out.append(1.0)
            else:
                out.append(0.0)
        return torch.tensor(out)
