import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from helper import get_statistics
from sumo_env import SumoEnv

n_actions = 2


def create_tls_model(n_observations=2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n_observations, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])
    return model


def choose_action(model, observation, single=True):
    # add batch dimension to the observation if only a single example was provided
    with tf.device(device_name):
        observation = np.expand_dims(observation, axis=0) if single else observation
        logits = model.predict(observation)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()
        action = tf.one_hot(action, n_actions, axis=-1)
        return action[0] if single else action


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


def discount_rewards(rewards, gamma=0.95):
    rewards -= np.mean(rewards, axis=0)
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)


def compute_loss(logits, actions, rewards):
    with tf.device(device_name):
        neg_logprob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        neg_logprob = tf.reshape(neg_logprob, rewards.shape)
        loss = tf.reduce_mean(neg_logprob * rewards)
        return loss


def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.device(device_name):
        with tf.GradientTape() as tape:
            logits = model(observations)
            loss = compute_loss(logits, actions, discounted_rewards)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss


def main(args):
    sumo_config_path = args[0] if len(args) > 0 else 'abstract_networks/grid/u_grid.sumocfg'
    multiple_detectors = (len(args) > 1 and args[1] == '--multiple-detectors')

    n_observations = 18 if multiple_detectors else 2
    tls_model = create_tls_model(n_observations)
    memory = Memory()

    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    env = SumoEnv(sumo_config_path, multiple_detectors)
    loss_history = []

    for i_episode in tqdm(range(500)):
        print('--------------------- Initializing epoch #', i_episode, '---------------------')
        if i_episode > 0:
            env.start_sumo()
        observation = env.get_observation()
        memory.clear()

        while True:
            action = choose_action(tls_model, observation, single=False)
            next_observation, reward, done = env.step(action)
            memory.add_to_memory(observation, action, reward)

            # is the episode over? did you crash or do so well that you're done?
            if done:

                # initiate training - remember we don't know anything about how the
                #   agent is doing until it has crashed!
                loss = train_step(tls_model, optimizer,
                           observations=np.vstack(memory.observations),
                           actions=memory.actions,
                           discounted_rewards=discount_rewards(memory.rewards, 0.95))
                loss_history.append(loss)
                memory.clear()
                break

            observation = next_observation

        env.reset()
        waiting_time_array = get_statistics()[0]
        print("Max: ", max(waiting_time_array))
        print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))

    if multiple_detectors:
        model_file_name = f'saved_models/DQL/multi_DQL_{time.strftime("%d.%m.%Y-%H:%M")}.h5'
    else:
        model_file_name = f'saved_models/DQL/DQL_{time.strftime("%d.%m.%Y-%H:%M")}.h5'
    tls_model.save(model_file_name)
    return loss_history


if __name__ == '__main__':
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        device_name = '/cpu:0'

    main(sys.argv[1:])
