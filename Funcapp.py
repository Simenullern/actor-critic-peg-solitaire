import torch
import numpy as np


class Funcapp:
    def __init__(self, critic, layers):
        modules = []

        for i in range(0, len(layers) - 1):
            modules.append(torch.nn.Linear((layers[i]), layers[i + 1], bias=True))
            critic.eligs.append(np.zeros(modules[i].weight.shape))
        modules.append(torch.nn.Sigmoid())
        critic.eligs = np.array(critic.eligs)

        self.funcapp = torch.nn.Sequential(*modules).train()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.funcapp.parameters(), lr=critic.learning_rate)
