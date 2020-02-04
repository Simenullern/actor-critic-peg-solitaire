import random

class GameController:
    def __init__(self, board, visualize=False):
        self.board = board
        self.visualize = visualize
        self.states = []
        self.actions = []

    def initialize_board(self):
        self.board.init_board()
        self.states.append(self.get_game_state())
        #self.actions.append((('init'), ('init')))
        self.show_board()

    def set_visualization(self, bool):
        self.visualize = bool

    def new_game(self):
        self.states = []
        self.actions = []
        self.initialize_board()

    def get_game_state(self):
        return self.board.get_hashable_state()

    def show_board(self):
        if self.visualize:
            self.board.visualize()

    def get_valid_moves(self):
        return self.board.get_all_possible_moves()

    def get_random_move(self):
        valid_moves = self.get_valid_moves()
        if len(valid_moves) == 0:
            return -1
        random.shuffle(valid_moves)
        random_move = valid_moves[0][:-1]
        return random_move

    def make_move(self, move):
        for x in self.get_valid_moves():
            possible_move = x[:-1]
            if move == possible_move:
                self.board.make_move(move)
                self.actions.append(move)
                self.states.append(self.get_game_state())
                self.show_board()
                if self.game_is_won():
                    return 1
                #if not self.game_is_on():
                    #return -0.001
                return 0 # try different rewards? E.g. if all edge pieces are gone

    def get_states_in_episode(self):
        return self.states

    def get_actions_in_episode(self):
        return self.actions

    def game_is_won(self):
        return self.board.is_success()

    def game_is_on(self):
        return len(self.get_valid_moves()) > 0

    def get_remaining_pegs(self):
        return self.board.get_remaining_pegs()

