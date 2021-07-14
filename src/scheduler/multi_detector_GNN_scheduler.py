from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np
import torch

class MultiDetectorGNNSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, observation):
        self.observation = observation
        super(MultiDetectorGNNSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class MultiDetectorGNNScheduler(SchedulerInterface):
    def __init__(self, net, edge_index):
        self.net = net
        self.edge_index = edge_index

    def predict(self, info: MultiDetectorGNNSchedulerInfo):
        logits = self.net(info.observation, self.edge_index)
        action = torch.multinomial(logits, num_samples=1)
        action = action.flatten().item()
        print(action)
        
        return action
