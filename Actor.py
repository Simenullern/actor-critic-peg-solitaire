


class Actor:
    def __init__(self, learning_rate = 0.1, elig_decay_rate=0.9, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.elig_decay_rate = elig_decay_rate
        self.discount_factor = discount_factor
        self.sap_func = dict()
        self.eligs = dict()

    def update_policy(self, state, action, td_error):
        if (state, action) not in self.sap_func.keys():
            self.sap_func[(state, action)] = 0

        self.sap_func[(state, action)] += \
            self.learning_rate * td_error * self.get_elig(state, action)
        return self.sap_func[(state, action)]

    def set_policy(self, state, action, value):
        self.sap_func[(state, action)] = value

    def get_action_from_state(self, state):
        best_action = None
        max_score = 0
        candidates = []
        for sap in self.sap_func.keys():
            if sap[0] == state: # potential candidate
                candidates.append(sap)
                if self.sap_func[sap] > max_score:
                    best_action = sap[1]
        #if len(candidates) > 1:
            #print(candidates)
            #breakpoint()
        return best_action

    def set_elig(self, state, action, value):
        self.eligs[(state, action)] = value

    def get_elig(self, state, action):
        return self.eligs[(state, action)] if (state,action) in self.eligs.keys() \
            else 0

    def update_elig(self, state, action):
        if (state, action) not in self.eligs.keys():
            self.set_elig(state, action, 0)
        self.eligs[state] = self.discount_factor * self.elig_decay_rate * self.eligs[(state, action)]
        return self.eligs[state]




