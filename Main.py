from Board import Board
from GameController import GameController
from Actor import Actor
from Critic import Critic
import matplotlib.pyplot as plt

NUM_EPISODES = 1000

if __name__ == '__main__':

    board = Board(shape='triangle', size=5, open_start_cells=[(2, 1)])
    actor = Actor(learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9)
    critic = Critic(use_nn=False, learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9)
    pegs_remaining = []

    for episode in range(NUM_EPISODES):
        game_controller = GameController(board, visualize=False)
        game_controller.new_game()
        state = game_controller.get_game_state()
        action = actor.get_action_from_state(state)
        if action is None:
            action = game_controller.get_random_move()
            actor.set_policy(state, action, value=0)

        while game_controller.game_is_on():
            reward = game_controller.make_move(action)
            if game_controller.game_is_won():
                print('Congratulations, you won game no ', episode)
                pegs_remaining.append(1)
                break

            succ_state = game_controller.get_game_state()
            action_from_succ_state = actor.get_action_from_state(succ_state)
            if action_from_succ_state is None:
                action_from_succ_state = game_controller.get_random_move()
                actor.set_policy(succ_state, action_from_succ_state, value=0)
            else:
                print("Wants to use ", action_from_succ_state, " in ", succ_state)

            actor.set_elig(succ_state, action_from_succ_state, 1)
            state_value = critic.get_value(state)
            succ_state_value = critic.get_value(succ_state)
            TD_error = critic.compute_TD_error(reward, state_value, succ_state_value)
            critic.set_elig(state, value=1)

            no_of_steps_to_rewind = len(game_controller.get_actions_in_episode())
            for step in range(no_of_steps_to_rewind-1, -1, -1):
                state_in_episode = game_controller.get_states_in_episode()[step]
                action_in_episode = game_controller.get_actions_in_episode()[step]
                critic_val = critic.update_value_func(state_in_episode, TD_error)
                critic_elig = critic.update_elig(state_in_episode)
                actor_sap = actor.update_policy(state_in_episode, action_in_episode, TD_error)
                actor_elig = actor.update_elig(state_in_episode, action_in_episode)
                print(actor_sap) # Always 0. So it is not updating correctly.


            state = succ_state
            action = action_from_succ_state
            #breakpoint()


        pegs_remaining.append(game_controller.get_remaining_pegs())
        print("you lost with remaining pegs", game_controller.get_remaining_pegs())



    plt.plot(pegs_remaining)
    plt.show()

