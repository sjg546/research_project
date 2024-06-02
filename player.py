import math
class Player:

    def __init__(self, player_id, mean, stddev):
        self.player_id = player_id
        self.mean = mean
        self.std_dev = stddev
        self.p = [mean]
        self.w = [(1/(math.pow(stddev,2)))]
        self.mean_pi = mean
        self.delta = 10
    