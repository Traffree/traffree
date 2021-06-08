import numpy as np


class Info:
    def __init__(self, light_name, north, east, south, west):
        self.light_name = light_name
        self.north = north
        self.south = south
        self.west = west
        self.east = east

    def __str__(self):
        return f'{self.north} {self.east} {self.south} {self.west}'


class Scheduler:
    def __init__(self):
        self.name = ""

    @staticmethod
    def predict(info: Info):
        sum = (info.west + info.east + info.north + info.south) + 0.01
        probability = (info.north + info.south + 0.005) / sum
        print("probability", probability)
        return np.random.binomial(1, 1 - probability)


