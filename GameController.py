

class GameController:
    def __init__(self, board, visualize=False):
        self.board = board
        self.visualize = visualize
        self.states = []
        self.actions = []

    def initialize_board(self):
        self.board.init_board()
        self.states.append(self.get_game_state())
        self.actions.append((('init'), ('init')))
        self.show_board()

    def new_game(self):
        self.initialize_board()

    def get_game_state(self):
        return self.board.get_hashable_state()

    def show_board(self):
        if self.visualize:
            self.board.visualize()

    def get_valid_moves(self):
        return self.board.get_all_possible_moves()

    def make_move(self, move):
        # If valid move ??
        self.board.make_move(move)
        self.actions.append(move)
        self.states.append(self.get_game_state())
        self.show_board()
        return 10 if self.game_is_won() else 0  # try different rewards?

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

