class Info:
    def __init__(self, light_name, north, east, south, west):
        self.light_name = light_name
        self.north = north
        self.south = south
        self.west = west
        self.east = east

    def __str__(self):
        return f'{self.north} {self.east} {self.south} {self.west}'