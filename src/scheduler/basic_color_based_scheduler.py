from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np


class BasicColorBasedSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, red, green):
        self.red = red
        self.green = green
        # self.yellow = yellow
        super(BasicColorBasedSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class BasicColorBasedScheduler(SchedulerInterface):
    @staticmethod
    def predict(info: BasicColorBasedSchedulerInfo):
        all_count = (info.green + info.red) + 0.01
        probability = info.green / all_count

        # 0 means maintain green, 1 means change to red
        if probability > 0.8:
            return 0
        elif probability < 0.2:
            return 1
        return np.random.binomial(1, 1-probability)
