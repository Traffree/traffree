from .scheduler_interface import SchedulerInterface, SchedulerInfoInterface


class MultiDetectorNeatSchedulerInfo(SchedulerInfoInterface):
    def __init__(self, light_name, red, green):
        self.red = red
        self.green = green
        super(MultiDetectorNeatSchedulerInfo, self).__init__(light_name)

    def __str__(self):
        return f'{self.red} {self.green}'


class MultiDetectorNeatScheduler(SchedulerInterface):
    def __init__(self, net):
        self.net = net

    def predict(self, info: MultiDetectorNeatSchedulerInfo):
        arr = info.red + info.green
        return 0 if self.net.activate(tuple([x for sub_arr in arr for x in sub_arr]))[0] >= 0 else 1
