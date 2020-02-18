import torch
import numpy as np

class Critic:
    def __init__(self, layers, use_nn=False, learning_rate=0.1,
                 elig_decay_rate=0.9, discount_factor=0.9):

        self.use_nn = use_nn
        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.value_func = None if use_nn else dict()
        self.eligs = [] if use_nn else dict()

        # Everything below is to deal with the neural network as function approximator
        self.funcapp = None
        self.criterion = None
        self.optimizer = None
        if use_nn:
            modules = []
            for i in range(0, len(layers)-1):
                modules.append(torch.nn.Linear((layers[i]), layers[i+1], bias=True))
                self.eligs.append(np.zeros(modules[i].weight.shape))
            modules.append(torch.nn.Sigmoid())
            self.funcapp = torch.nn.Sequential(*modules).train()
            self.eligs= np.array(self.eligs)
            self.criterion = torch.nn.MSELoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.funcapp.parameters(), lr=self.learning_rate)

    def init_state_value_if_needed(self, state):
        if not self.use_nn and state not in self.value_func.keys():
            self.value_func[state] = 0

    def get_value(self, state):
        if self.use_nn:
            X = self.vectorize_state(state)
            return self.funcapp(X)
        else:
            self.init_state_value_if_needed(state)
            return self.value_func[state]

    def update_value_func(self, state, td_error, critic_elig):
        #print(critic_elig)
        #breakpoint()
        if self.use_nn:
            # backprop TD_error to get gradients, but wait with updating the weights
            X = Critic.vectorize_state(state)
            y_pred = self.funcapp(X)
            y_target = td_error + y_pred
            #elig_contribution = torch.tensor(critic_elig.flat[0], dtype=torch.float32)
            loss = self.criterion(y_pred, y_target)
            loss.backward(retain_graph = True)

            # Update eligs with standard partial derivative
            new_eligs = []
            for i in range(0, len(self.funcapp) - 1):
                gradients = self.funcapp[i].weight.grad.numpy()
                new_eligs.append(np.add(self.eligs[i], gradients))
            self.eligs = np.array(new_eligs)

            # Then update gradients with the elig contribution
            for i in range(0, len(self.funcapp) - 1):
                eligs = self.eligs[i]
                self.funcapp[i].weight.grad *= torch.tensor(eligs, dtype=torch.float)

            # Now the weights can be updated
            self.optimizer.step()

            return y_pred

        else:
            self.value_func[state] = self.value_func[state] + self.learning_rate * td_error * critic_elig
            return self.value_func[state]

    def get_elig(self, state):
        return self.eligs if self.use_nn else self.eligs[state]

    def set_elig(self, state, value):
        if self.use_nn:
            for layer in range(0, len(self.eligs)):
                for row in range(0, len(self.eligs[layer])):
                    for col in range(0, len(self.eligs[layer][row])):
                        self.eligs[layer][row][col] = value
        else:
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
