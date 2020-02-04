from Funcapp import Funcapp
import torch
import numpy as np

class Critic:
    def __init__(self, layers, use_nn=False, learning_rate_nn=0.01, learning_rate=0.1,
                 elig_decay_rate=0.9, discount_factor=0.9):

        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.value_func = dict()
        self.eligs = dict()
        self.use_nn = use_nn
        self.funcapp = None
        self.nn_learning_rate = None
        self.elig_of_weights = []
        criterion = None
        optimizer = None
        if use_nn:
            modules = []
            for i in range(0, len(layers)-1):
                modules.append(torch.nn.Linear((layers[i]), layers[i+1]))
                self.elig_of_weights.append(np.zeros(modules[i].weight.shape))  # zeros
            self.funcapp = torch.nn.Sequential(*modules)
            self.funcapp.train()
            self.nn_learning_rate = learning_rate_nn
            self.criterion = torch.nn.MSELoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.funcapp.parameters(), lr=self.nn_learning_rate)

        ## Eligbs first have to be initialized to the size of all the layers of weights. E.g just use random weight init.

        ##For each state, action thing:
            # Update all eligs with the standard elig
            # Get y_pred and y_target
            ## Then propegate loss and get gradients
            ## Then update eligs again with gradients


        ## Final policy. model.eval()


    def init_state_value_if_needed(self, state):
        if state not in self.value_func.keys():
            self.value_func[state] = 0

    def get_value(self, state):
        self.init_state_value_if_needed(state)
        return self.value_func[state]

    def update_value_func(self, state, td_error, critic_elig):
        if self.use_nn:
            X = self.vectorize_state(state)
            #breakpoint()
            y_pred = self.funcapp(X)
            #breakpoint()
            self.value_func[state] = y_pred
            #breakpoint()

            y_target = td_error + y_pred ## Is this correct?
            #breakpoint()

            #breakpoint()
            loss = self.criterion(y_pred, y_target)  # basically just the TD ERROR
            #breakpoint()
            loss.backward(retain_graph = True)
            self.optimizer.step()

            ## Update eligs
            new_eligs = []
            for i in range(0, len(self.funcapp)):
                gradients = self.funcapp[i].weight.grad.numpy()
                new_eligs.append(np.add(self.elig_of_weights[i], gradients))
            self.elig_of_weights = np.array(new_eligs)


        else:
            self.value_func[state] = self.value_func[state] + self.learning_rate * td_error * critic_elig
        return self.value_func[state]

    def vectorize_state(self, state):
        out = []
        for char in state:
            if char == '1':
                out.append(1.0)
            else:
                out.append(0.0)
        return torch.tensor(out)


    def get_elig(self, state):
        return self.eligs[state]

    def set_elig(self, state, value):
        if self.use_nn:
            for i in range(0, len(self.elig_of_weights)):
                self.elig_of_weights[i].fill(value)
        self.eligs[state] = value

    def update_elig(self, state, critic_elig):
        if self.use_nn:
            for i in range (0, len(self.elig_of_weights)):
                self.elig_of_weights[i].fill(self.discount_factor * self.elig_decay_rate * critic_elig)

        self.eligs[state] = self.discount_factor * self.elig_decay_rate * critic_elig
        return self.eligs[state]

    def reset_eligs(self):
        if self.use_nn:
            self.set_elig(None, 0)
        for key in self.eligs.keys():
            self.eligs[key] = 0

    def compute_TD_error(self, reward, state_value, succ_state_value):
        return reward + self.discount_factor * succ_state_value - state_value

    def get_target_for_state_evalutation(self, reward, succ_state_value):
        ## LOSS Is the TD error.
        ## Eligibilites must be updates getting the gradient as well

        # The key assumption behind temporal differencing is that the correct value for V(s) is r +γV(s')
        # where s’ is the next state and r is the reward received in going from s to s’).
        return reward + self.discount_factor*succ_state_value
