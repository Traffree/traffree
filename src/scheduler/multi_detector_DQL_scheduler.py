import tensorflow as tf
from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface
import numpy as np


class MultiDetectorDQLSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, red, green):
        self.red = red
        self.green = green
        super(MultiDetectorDQLSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class MultiDetectorDQLScheduler(SchedulerInterface):
    def __init__(self, net):
        self.net = net

    def predict(self, info: MultiDetectorDQLSchedulerInfo):
        arr = info.red + info.green
        observation = np.array([x for sub_arr in arr for x in sub_arr])
        observation = np.expand_dims(observation, axis=0)
        logits = self.net(observation)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()[0]

        return action
