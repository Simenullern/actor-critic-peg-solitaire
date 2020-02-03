from Board import Board
from GameController import GameController
from Actor import Actor
from Critic import Critic
import matplotlib.pyplot as plt
import time

NUM_EPISODES = 500
VERBOSE_GAME_OUTCOME = True
VISUALIZE_FINAL_TARGET_POLICY = False
SLEEP_BETWEEN_MOVES = 0

BOARD_SHAPE = 'triangle'
BOARD_SIZE = 5
OPEN_START_CELLS = [(2, 1)]
VISUALIZE_ALL_GAMES = False

if __name__ == '__main__':

    board = Board(shape=BOARD_SHAPE, size=BOARD_SIZE, open_start_cells=OPEN_START_CELLS)
    game_controller = GameController(board, visualize=VISUALIZE_ALL_GAMES)
    actor = Actor(learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9, epsilon=0.5,
                  random_move_generator=game_controller.get_random_move)
    critic = Critic(use_nn=False, learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9)
    pegs_remaining = []

    for episode in range(NUM_EPISODES):
        if VISUALIZE_FINAL_TARGET_POLICY and episode == NUM_EPISODES - 1:
            actor.set_episilon(0)
            game_controller.set_visualization(True)
            SLEEP_BETWEEN_MOVES = 0.5

        game_controller.new_game()
        state = game_controller.get_game_state()
        action = actor.get_action_from_state(state)

        while game_controller.game_is_on():
            time.sleep(SLEEP_BETWEEN_MOVES)
            critic.reset_eligs()
            actor.reset_eligs()
            critic.init_state_value_if_needed(state)
            actor.init_sap_value_if_needed(state, action)

            reward = game_controller.make_move(action)

            succ_state = game_controller.get_game_state()
            action_from_succ_state = actor.get_action_from_state(succ_state)

            actor.set_elig(state, action, value=1)
            critic.set_elig(state, value=1)

            state_value = critic.get_value(state)
            succ_state_value = critic.get_value(succ_state)
            TD_error = critic.compute_TD_error(reward, state_value, succ_state_value)

            critic_val = critic.get_value(state)
            critic_elig = critic.get_elig(state)
            actor_val = actor.get_sap_value(state, action)
            actor_elig = actor.get_elig(state, action)

            no_of_steps_to_rewind = len(game_controller.get_actions_in_episode())

            for move in range(no_of_steps_to_rewind-1, -1, -1):
                state_in_episode = game_controller.get_states_in_episode()[move]
                action_in_episode = game_controller.get_actions_in_episode()[move]

                ## UPDATES
                critic_elig = critic.update_elig(state_in_episode, critic_elig)
                actor_elig = actor.update_elig(state_in_episode, action_in_episode, actor_elig)
                critic_val = critic.update_value_func(state_in_episode, TD_error, critic_elig)
                actor_val = actor.update_policy(state_in_episode, action_in_episode, TD_error, actor_elig)

            state = succ_state
            action = action_from_succ_state

        if game_controller.game_is_won():
            if VERBOSE_GAME_OUTCOME: print('Congratulations, you won game no ', episode)
            pegs_remaining.append(1)
            actor.set_episilon(actor.get_episilon()*0.9)
        else:
            if VERBOSE_GAME_OUTCOME: print("you lost with remaining pegs", game_controller.get_remaining_pegs())
            pegs_remaining.append(game_controller.get_remaining_pegs())



    ## PLOT RESULTS
    plt.plot(pegs_remaining)
    plt.ylabel("Number of pegs remaining")
    plt.xlabel("Number of episodes")
    plt.show()

