import random

class Actor:
    def __init__(self, random_move_generator, learning_rate = 0.1, elig_decay_rate=0.9, discount_factor=0.9,
                 epsilon = 0.1):
        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.random_move_generator = random_move_generator
        self.sap_func = dict()
        self.eligs = dict()

    def init_sap_if_needed(self, state, action):
        if (state, action) not in self.sap_func.keys():
            self.sap_func[(state, action)] = 0

    def update_policy(self, state, action, actor_val, td_error):
        #if (state, action) not in self.sap_func.keys():
            #self.sap_func[(state, action)] = 0

        self.sap_func[(state, action)] = self.sap_func[(state, action)] + self.learning_rate * td_error * self.get_elig(state, action)

        return self.sap_func[(state, action)]

    def get_sap_value(self, state, action):
        return self.sap_func[(state, action)]

    def set_policy(self, state, action, value):
        self.sap_func[(state, action)] = value

    def get_action_from_state(self, state):
        if random.uniform(0, 1) < self.epsilon:
            print("\tchose random val because of epsillon!")
            return self.random_move_generator()

        best_action = None
        max_score = 0
        candidates = []
        for sap in self.sap_func.keys():
            if sap[0] == state: # potential candidate
                candidates.append(sap)
                if self.sap_func[sap] > max_score:
                    best_action = sap[1]
        #if len(candidates) > 1:
            #print(candidates, self.sap_func[candidates[0]], self.sap_func[candidates[0]])
            #breakpoint()
        if best_action is None:
            #print("\tchose random val because no candidates exist!")
            return self.random_move_generator()
        else:
            #print("\tchose best action")
            return best_action

    def set_elig(self, state, action, value):
        self.eligs[(state, action)] = value

    def get_elig(self, state, action):
        return self.eligs[(state, action)] if (state,action) in self.eligs.keys() \
            else 0

    def update_elig(self, state, action, actor_elig):
        if (state, action) not in self.eligs.keys():
            self.set_elig(state, action, 0)
        self.eligs[state] = self.discount_factor * self.elig_decay_rate * actor_elig
        return self.eligs[state]

    def reset_eligs(self):
        for key in self.eligs.keys():
            self.eligs[key] = 0




