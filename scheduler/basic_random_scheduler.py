from scheduler.scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np


class BasicRandomSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, north, east, south, west):
        self.north = north
        self.south = south
        self.west = west
        self.east = east
        super(BasicRandomSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.north} {self.east} {self.south} {self.west}'


class BasicRandomScheduler(SchedulerInterface):
    @staticmethod
    def predict(info: SchedulerInfoInterface):
        all_count = (info.west + info.east + info.north + info.south) + 0.01
        probability = (info.north + info.south + 0.005) / all_count
        return np.random.binomial(1, 1 - probability)
