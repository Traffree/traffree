class SchedulerInfoInterface:
    def __init__(self, light_name):
        self.light_name = light_name


class SchedulerInterface:
    @staticmethod
    def predict(info: SchedulerInfoInterface):
        pass


