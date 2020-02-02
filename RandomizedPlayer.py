from Board import Board
from GameController import GameController
import random
import matplotlib.pyplot as plt
import time





if __name__ == '__main__':
    board = Board(shape='triangle', size=5, open_start_cells=[(2, 2)])
    controller = GameController(board, visualize=False)

    num_episodes = 100
    pegs_remaining = []

    for episode in range(num_episodes):
        controller.new_game()
        controller.show_board()

        while not controller.game_is_lost():
            if controller.game_is_won():
                print('Congratulations, you won game no ', episode)
                pegs_remaining.append(1)
                break
            valid_moves = controller.get_valid_moves()
            random.shuffle(valid_moves)
            random_move = valid_moves[0][:-1]
            reward = controller.make_move(random_move)
            #time.sleep(1)
            controller.show_board()
            if controller.game_is_lost():
                print("You lost game no ", episode)
                pegs_remaining.append(controller.get_remaining_pegs())

    plt.plot(pegs_remaining)
    plt.show()

