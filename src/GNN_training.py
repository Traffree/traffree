import time
from datetime import datetime

# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import sumolib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import traci
from tqdm import tqdm

from helper import get_edge_index, get_node_to_index, get_statistics
from sumo_env import SumoEnv


class GNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim=32, dropout=0.25):
        super(GNNModel, self).__init__()

        # dim = [num_tls, num_features]
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim))
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # dim = [num_tls, hidden_dim]
        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # [num_tls, hidden_dim]
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.post_mp(x)

        return F.softmax(x, dim=1)


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


# Helper function to combine a list of Memory objects into a single Memory.
#     This will be very useful for batching.
def aggregate_memories(memories):
    batch_memory = Memory()

    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)

    return batch_memory


def choose_action(model, observation, edge_index, n_actions=2):
    # add batch dimension to the observation if only a single example was provided
    with torch.no_grad():
        logits = model(observation, edge_index)
        action = torch.multinomial(logits, num_samples=1)
        action = action.flatten()
        action = F.one_hot(action, n_actions)
        return action


def compute_loss(actions, rewards, q, q_next, gamma=0.95):
    rewards = torch.FloatTensor(rewards)
    actions = actions.float()
    loss = rewards + gamma * torch.min(q_next, dim=1)[0] - (q * actions).sum(dim=1)
    loss = torch.square(loss)
    return loss.sum() / loss.shape[0]


def train_step(model, optimizer, observations, edge_index, actions, rewards):
    for action, reward, observation, next_observation in zip(
        actions, rewards, observations[:-1], observations[1:]
    ):
        q = model(observation, edge_index)
        with torch.no_grad():
            q_next = model(next_observation, edge_index)
        loss = compute_loss(action, reward, q, q_next)
        loss.backward()
        optimizer.step()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    sumo_config_path = (
        args[0] if len(args) > 0 else "scenarios/medium_grid/normal/config.sumocfg"
    )
    multiple_detectors = len(args) > 1 and args[1] == "--multiple-detectors"
    net_file = "scenarios/small_grid/map.net.xml"
    net = sumolib.net.readNet(net_file)
    edge_index = torch.LongTensor(get_edge_index(net).T)

    num_features = 18 if multiple_detectors else 2
    model = GNNModel(
        input_dim=num_features, output_dim=2, num_layers=1, dropout=0.25
    ).to(device)

    memory = Memory()

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sumo_env = SumoEnv(sumo_config_path, multiple_detectors)

    for i_episode in tqdm(range(1)):
        print(f"--------------------- Initializing epoch #{i_episode}---------------------")
        if i_episode > 0:
            sumo_env.start_sumo()
        observation = torch.FloatTensor(sumo_env.get_observation())
        memory.clear()

        while True:
            action = choose_action(model, observation, edge_index)
            next_observation, reward, done = sumo_env.step(action)
            memory.add_to_memory(observation, action, reward)

            # is the episode over? did you crash or do so well that you're done?
            if done:
                # initiate training - remember we don't know anything about how the
                #   agent is doing until it has crashed!
                train_step(
                    model,
                    optimizer,
                    observations=memory.observations,
                    edge_index=edge_index,
                    actions=memory.actions,
                    rewards=memory.rewards,
                )

                memory.clear()
                break

            observation = torch.FloatTensor(next_observation)

        sumo_env.reset()
        waiting_time_array = get_statistics()[0]
        print("Max: ", max(waiting_time_array))
        print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))

    # if multiple_detectors:
    #     model_file_name = f'saved_models/GCNN/multi_GCNN_{time.strftime("%d.%m.%Y-%H:%M")}.h5'
    # else:
    #     model_file_name = f'saved_models/GCNN/GCNN_{time.strftime("%d.%m.%Y-%H:%M")}.h5'
    # model.save(model_file_name)


if __name__ == "__main__":
    main(["scenarios/small_grid/normal/config.sumocfg", "--multiple-detectors"])
