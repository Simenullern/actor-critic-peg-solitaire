


class Actor:
    def __init__(self, game_controller, use_nn=False, learning_rate=0.1, decay_rate=0.1):
        self.game_controller = game_controller
        self.use_nn = use_nn
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.sap_func = dict()
        self.eligs = dict()

