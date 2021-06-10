from scheduler.scheduler_interface import SchedulerInterface
from scheduler.info import Info
import numpy as np


class BasicRandomScheduler(SchedulerInterface):
    @staticmethod
    def predict(info: Info):
        all_count = (info.west + info.east + info.north + info.south) + 0.01
        probability = (info.north + info.south + 0.005) / all_count
        return np.random.binomial(1, 1 - probability)
