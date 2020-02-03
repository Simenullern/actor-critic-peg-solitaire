from Board import Board
from GameController import GameController
from Actor import Actor
from Critic import Critic
import matplotlib.pyplot as plt

NUM_EPISODES = 1000
PRINT = True

if __name__ == '__main__':

    board = Board(shape='triangle', size=5, open_start_cells=[(2, 1)])
    game_controller = GameController(board, visualize=False)
    actor = Actor(learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9, epsilon=0.1,
                  random_move_generator=game_controller.get_random_move)
    critic = Critic(use_nn=False, learning_rate=0.3, elig_decay_rate=0.75, discount_factor=0.9)
    pegs_remaining = []

    for episode in range(NUM_EPISODES):
        game_controller.new_game()
        state = game_controller.get_game_state()
        action = actor.get_action_from_state(state)

        while game_controller.game_is_on():
            critic.reset_eligs()
            actor.reset_eligs()
            critic.init_value_if_needed(state)
            actor.init_sap_if_needed(state, action)

            reward = game_controller.make_move(action)

            succ_state = game_controller.get_game_state()
            action_from_succ_state = actor.get_action_from_state(succ_state)

            actor.set_elig(state, action, value=1)
            critic.set_elig(state, value=1)

            state_value = critic.get_value(state)
            succ_state_value = critic.get_value(succ_state)
            TD_error = critic.compute_TD_error(reward, state_value, succ_state_value)

            no_of_steps_to_rewind = len(game_controller.get_actions_in_episode())

            #print("Rewinding", no_of_steps_to_rewind, "from", state)
            critic_val = critic.get_value(state)
            critic_elig = critic.get_elig(state)
            actor_val = actor.get_sap_value(state, action)
            actor_elig = actor.get_elig(state, action)

            for move in range(no_of_steps_to_rewind-1, -1, -1):
                state_in_episode = game_controller.get_states_in_episode()[move]
                action_in_episode = game_controller.get_actions_in_episode()[move]

                print("Considering ", state_in_episode, action_in_episode)
                print("critic val", critic_val, "critic_elig", critic_elig, "actor val", actor_val, "actor_elig", actor_elig)

                ## Updates
                critic_elig = critic.update_elig(state_in_episode, critic_elig)
                critic_val = critic.update_value_func(state_in_episode, critic_val, TD_error)
                actor_elig = actor.update_elig(state_in_episode, action_in_episode, actor_elig)
                actor_val = actor.update_policy(state_in_episode, action_in_episode, actor_val,  TD_error)

                #print("now updating")
                print("After updating: critic val", critic_val, "critic elig", critic_elig,
                      "actor_val", actor_val, "actor_elig", actor_elig) # A
                breakpoint()

                if game_controller.game_is_won():
                    breakpoint()



            state = succ_state
            action = action_from_succ_state

        if game_controller.game_is_won():
            # breakpoint()
            print('Congratulations, you won game no ', episode)
            pegs_remaining.append(1)
        else:
            pegs_remaining.append(game_controller.get_remaining_pegs())
            print("you lost with remaining pegs", game_controller.get_remaining_pegs())



    plt.plot(pegs_remaining)
    plt.show()

