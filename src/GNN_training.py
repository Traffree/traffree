import pickle
import time
from datetime import datetime
import random

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

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []

    def add_to_memory(self, observation, action, reward, next_observation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
    
    def sample(self, k=0.4):
        l = len(self.observations)
        sample = random.sample(range(l), int(l * k))
        sampled_memory = Memory()
        for i in sample:
            sampled_memory.add_to_memory(
                self.observations[i],
                self.actions[i],
                self.rewards[i],
                self.next_observations[i]
            )
        return sampled_memory



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
    loss = rewards + gamma * torch.min(q_next, dim=1)[0] - (q * actions).sum(dim=1)
    loss = torch.square(loss)
    return loss.sum() / loss.shape[0]


def train_step(model, optimizer, observations, next_observations, edge_index, actions, rewards):
    observations = torch.FloatTensor(observations).to(device)
    next_observations = torch.FloatTensor(next_observations).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    actions = torch.stack(actions).to(device)
    total_loss = 0
    for action, reward, observation, next_observation in zip(
        actions, rewards, observations, next_observations
    ):
        q = model(observation, edge_index)
        with torch.no_grad():
            q_next = model(next_observation, edge_index)
        loss = compute_loss(action, reward, q, q_next)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('CURRENT LOSS:', total_loss)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gnn_model(
    sumo_config_path="abstract_networks/grid/u_grid.sumocfg",
    net_file="abstract_networks/grid/u_grid.net.xml",
    multiple_detectors=True,
    num_epochs=50,
    memory_size=30
):
    num_features = 18 if multiple_detectors else 2

    net = sumolib.net.readNet(net_file)
    edge_index = torch.LongTensor(get_edge_index(net).T).to(device)

    model = GNNModel(
        input_dim=num_features,
        output_dim=2,
        num_layers=1,
        dropout=0.25
    ).to(device)

    memory = Memory()

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sumo_env = SumoEnv(sumo_config_path, multiple_detectors)

    for i_episode in tqdm(range(num_epochs)):
        print(f"--------------------- Initializing epoch #{i_episode}---------------------")
        if i_episode > 0:
            sumo_env.start_sumo()
        
        memory.clear()
        prev_observation = torch.FloatTensor(sumo_env.get_observation()).to(device)
        prev_action = 0
        prev_reward = 0
        k = 1
        while True:
            action = choose_action(model, prev_observation, edge_index)
            observation, reward, done = sumo_env.step(action)
            if done:
                break

            observation = torch.FloatTensor(observation).to(device)
            if k > 1:
                memory.add_to_memory(prev_observation, prev_action, prev_reward, observation)
            
            if k % memory_size == 0:
                sampled_memory = memory.sample()
                train_step(
                    model,
                    optimizer,
                    observations=sampled_memory.observations,
                    next_observations=sampled_memory.next_observations,
                    edge_index=edge_index,
                    actions=sampled_memory.actions,
                    rewards=sampled_memory.rewards,
                )
                memory.clear()
            prev_action = action
            prev_reward = reward
            prev_observation = observation
            k += 1

        sumo_env.reset()
        waiting_time_array = get_statistics()[0]
        print("Max: ", max(waiting_time_array))
        print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))

    if multiple_detectors:
        model_file_name = f'saved_models/GNN/multi_GNN_{time.strftime("%d.%m.%Y-%H:%M")}.pt'
    else:
        model_file_name = f'saved_models/GNN/GNN_{time.strftime("%d.%m.%Y-%H:%M")}.pt'
    torch.save(model.state_dict(), model_file_name)


def train_gnn_model_offline(
        net_file="scenarios/medium_grid/u_map.net.xml",
        multiple_detectors=True,
        num_epochs=50,
        memory_path='scenarios/medium_grid/training/memory.pkl',
        k=50,
):
    num_features = 18

    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    print(f"Memory consists of {len(memory)} observations")
    
    net = sumolib.net.readNet(net_file)
    edge_index = torch.LongTensor(get_edge_index(net).T).to(device)

    model = GNNModel(
        input_dim=num_features,
        output_dim=2,
        num_layers=1,
        dropout=0.25
    ).to(device)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i_episode in tqdm(range(num_epochs)):
        print(f"--------------------- Initializing epoch #{i_episode}---------------------")
        sampled_memory = memory.sample(k / len(memory))
        train_step(
            model,
            optimizer,
            observations=sampled_memory.observations,
            next_observations=sampled_memory.next_observations,
            edge_index=edge_index,
            actions=sampled_memory.actions,
            rewards=sampled_memory.rewards,
        )

    if multiple_detectors:
        model_file_name = f'saved_models/GNN/multi_GNN_offline_{time.strftime("%d.%m.%Y-%H:%M")}.pt'
    else:
        model_file_name = f'saved_models/GNN/GNN_offline_{time.strftime("%d.%m.%Y-%H:%M")}.pt'
    torch.save(model.state_dict(), model_file_name)


if __name__ == "__main__":
    # train_gnn_model()
    
    train_gnn_model_offline(num_epochs=100)        

