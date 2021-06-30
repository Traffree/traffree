import pickle

import tensorflow as tf

from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np


class DQLSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, red, green):
        self.red = red
        self.green = green
        super(DQLSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class DQLScheduler(SchedulerInterface):
    def __init__(self, net):
        self.net = net

    def predict(self, info: DQLSchedulerInfo):
        observation = np.array([info.red, info.green])
        observation = np.expand_dims(observation, axis=0)
        logits = self.net.predict(observation)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()[0]

        # print('----- Action:', action)
        return action
