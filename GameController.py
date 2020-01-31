

class GameController:
    def __init__(self, board, visualize=False):
        self.board = board
        self.visualize = visualize

        self.initialize_board()

    def initialize_board(self):
        self.board.init_board()

    def new_game(self):
        self.initialize_board()

    def show_board(self):
        if self.visualize:
            self.board.visualize()

    def get_valid_moves(self):
        return self.board.get_all_possible_moves()

    def make_move(self, move):
        self.board.make_move(move)

    def game_is_won(self):
        return self.board.is_success()

    def game_is_lost(self):
        return len(self.get_valid_moves()) == 0 and not self.game_is_won()

    def get_remaining_pegs(self):
        return self.board.get_remaining_pegs()

