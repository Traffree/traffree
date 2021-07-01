import pickle

import neat

from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np


class NeatSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, red, green):
        self.red = red
        self.green = green
        super(NeatSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class NeatScheduler(SchedulerInterface):
    def __init__(self, net):
        self.net = net

    def predict(self, info: NeatSchedulerInfo):
        return 0 if self.net.activate((info.red, info.green))[0] >= 0 else 1
