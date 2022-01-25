from project_utils import DDPG, environment
import argparse
import tensorflow as tf
import numpy as np
import pystk
from tensorflow.keras.utils import Progbar

def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    legal_action[1] = np.clip(legal_action[1], 0, 1)
    legal_action[2:] = legal_action[2:] > 0

    return [np.squeeze(legal_action)]

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gamma", type=float,
        default=0.99, help="discount factor")
    ap.add_argument("-s", "--num_steps", type=int,
        default=1000, help="maximum number of steps per episode")
    ap.add_argument("-e", "--num_episodes", type=int,
        default=10000, help="maximum number of episodes")
    ap.add_argument("--std_dev", type=float, default=0.2,
        help="standard deviation of perturbation noise")
    ap.add_argument("-cl", "--critic_learning_rate", type=float,
        help="critic model learning rate", default=0.002)
    ap.add_argument("-al", "--actor_learning_rate", type=float,
        help="actor model learning rate", default=0.001)
    args = ap.parse_args()
    return args

def main():
    max_iters = 5000

    env = environment.PyTuxActionCritic(verbose=True, steps=max_iters)
    num_states = env.config.screen_width * env.config.screen_height * 3
    num_actions = 7

    upper_bound = 1
    lower_bound = -1

    std_dev = 0.3 * np.ones(num_actions)
    # std_dev[std_dev > 0.5] = 0.5
    ou_noise = DDPG.OUActionNoise(mean=np.zeros(num_actions), std_deviation=std_dev)

    actor_model = DDPG.get_actor(num_actions, num_states, upper_bound)
    critic_model = DDPG.get_critic(num_states, num_actions)

    target_actor = DDPG.get_actor(num_actions, num_states, upper_bound)
    target_critic = DDPG.get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.02
    actor_lr = 0.01

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # critic_optimizer = tf.keras.optimizers.SGD(critic_lr, momentum=0.9)
    # actor_optimizer = tf.keras.optimizers.SGD(actor_lr, momentum=0.9)

    total_episodes = 100000
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    model_dict = {
        'target_actor': target_actor,
        'target_critic': target_critic,
        'critic_model': critic_model,
        'actor_model': actor_model,
        'critic_optimizer': critic_optimizer,
        'actor_optimizer': actor_optimizer
    }

    buffer = DDPG.Buffer(num_states, num_actions, model_dict, 
        gamma=gamma, buffer_capacity=1000)

    del model_dict
    del actor_model
    del critic_model
    del target_critic
    del target_actor

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    tracks = ['hacienda']
    # Takes about 4 min to train

    for track in tracks:
        for ep in range(1, total_episodes+1):
            env.restart(track)
            print("\nRunning episode: {}/{}".format(ep, total_episodes))

            prev_state = env.restart(track)
            prev_state = prev_state.flatten()
            episodic_reward = 0

            bar = Progbar(max_iters, stateful_metrics=['reward', 'distance'])

            for i in range(max_iters):
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = policy(tf_prev_state, ou_noise, 
                    buffer.model_dict['actor_model'], lower_bound, upper_bound)

                # Recieve state and reward from environment.
                state, reward, done, dist = env.step(pystk.Action(*(
                    action[0].tolist())))

                state = state.flatten()

                buffer.record((prev_state, action[0], reward, state))
                episodic_reward += reward

                buffer.learn()
                DDPG.update_target(buffer.model_dict['target_actor'].variables, buffer.model_dict['actor_model'].variables, tau)
                DDPG.update_target(buffer.model_dict['target_critic'].variables, buffer.model_dict['critic_model'].variables, tau)

                # End this episode when `done` is True
                if done:
                    bar.add(max_iters-i, values=[('reward', episodic_reward),
                        ('distance', dist)])
                    break

                prev_state = state

                # model_dict = {
                #     'target_actor': target_actor,
                #     'target_critic': target_critic,
                #     'critic_model': critic_model,
                #     'actor_model': actor_model,
                #     'critic_optimizer': critic_optimizer,
                #     'actor_optimizer': actor_optimizer
                # }

                # buffer.updateDict(model_dict)

                bar.add(1, values=[('reward', episodic_reward),
                        ('distance', dist)])
        
            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("\nEpisode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

if __name__ == "__main__":
    main()


