from Board import Board
from GameController import GameController
from Critic import Critic
import random

NUM_EPISODES = 100

if __name__ == '__main__':

    board = Board(shape='triangle', size=5, open_start_cells=[(2, 1)])
    critic = Critic(use_nn=False, learning_rate=0.1, elig_decay_rate=0.9, discount_factor=0.9)
    pegs_remaining = []

    for episode in range(NUM_EPISODES):
        game_controller = GameController(board, visualize=False)
        game_controller.new_game()
        state = game_controller.get_game_state()

        # Just start with an actor doing random moves at first move

        while game_controller.game_is_on():
            # Actor doing random move for now
            valid_moves = game_controller.get_valid_moves()
            random.shuffle(valid_moves)
            random_move = valid_moves[0][:-1]
            reward = game_controller.make_move(random_move)
            succ_state = game_controller.get_game_state()

            if game_controller.game_is_won():
                print('Congratulations, you won game no ', episode)
                pegs_remaining.append(1)
                break

            state_value = critic.get_value(state)
            succ_state_value = critic.get_value(succ_state)
            TD_error = critic.compute_TD_error(reward, state_value, succ_state_value)
            critic.set_elig(state, value=1)

            no_of_steps_to_rewind = len(game_controller.get_actions_in_episode())
            for step in range(no_of_steps_to_rewind-1, -1, -1):
                state_in_episode = game_controller.get_states_in_episode()[step]
                action_in_episode = game_controller.get_actions_in_episode()[step]
                val = critic.update_value_func(state_in_episode, TD_error)
                elig = critic.update_elig(state_in_episode)

                ## Actor updates

            state = succ_state

        pegs_remaining.append(game_controller.get_remaining_pegs())
        print("you lost with remaining pegs", game_controller.get_remaining_pegs())
        #breakpoint()







    # Then print and see that critics updates are correct
    # Then move into making the actor

