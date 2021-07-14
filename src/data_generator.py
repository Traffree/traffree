import os
import pickle
import time

import tensorflow as tf
# from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from tqdm import tqdm

from GNN_training import Memory
from helper import get_statistics
from sumo_env import SumoEnv


def generate_GNN_data(path_to_dir, model_file):
    configs = []
    for file in os.listdir(path_to_dir):
        if '.sumocfg' in file:
            configs.append(f'{path_to_dir}/{file}')

    memory = Memory()
    net = tf.keras.models.load_model(model_file, compile=False)

    for config in tqdm(configs):
        scenario_name = config.split('/')[-1]
        print(f"--------------------- Simulating scenario: {scenario_name} ---------------------")

        sumo_env = SumoEnv(config, multiple_detectors=True)

        prev_observation = sumo_env.get_observation()
        prev_action = 0
        prev_reward = 0
        valid = False

        while True:
            logits = net(prev_observation)
            logits = logits.numpy()
            logits = torch.from_numpy(logits)
            logits = F.softmax(logits, dim=1)
            action = torch.multinomial(logits, num_samples=1)
            action = action.flatten()
            action = F.one_hot(action, 2)

            observation, reward, done = sumo_env.step(action)
            if done:
                break

            if valid:
                memory.add_to_memory(prev_observation, prev_action, prev_reward, observation)
            else:
                valid = True

            prev_action = action
            prev_reward = reward
            prev_observation = observation

        sumo_env.reset()
        waiting_time_array = get_statistics()[0]
        print("Max: ", max(waiting_time_array))
        print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))

    memory_file_name = f'scenarios/medium_grid/training/memory.pkl'
    with open(memory_file_name, "wb") as f:
        pickle.dump(memory, f)
        f.close()


if __name__ == "__main__":
    generate_GNN_data('scenarios/medium_grid/training', 'saved_models/DQL/multi_DQL_02.07.2021-16:04.h5')
