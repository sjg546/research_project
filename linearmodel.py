import numpy as np
import math
class LinearModel():
    def __init__(self):
        self._intercept = 0.5
        self._slope = 0.000565
        self._match_count = 0
        self._log_loss_list = []

    def predict_match(self, winner_points, loser_points):
        diff_points = winner_points - loser_points        
        probability = self._calculate_probability(diff_points)
        if probability > 0.5:
            self._log_loss_list.append(self.log_loss(1.0,probability))
        else:
            self._log_loss_list.append(self.log_loss(0.0, probability))

        return self._predict(probability)
    
    def _calculate_probability(self,diff_points):
        probability = self._intercept + (self._slope *diff_points)
        return probability
    
    def _predict(self,probability):
        random_prob = np.random.uniform(0,1)
        if random_prob < probability:
            return True
        else:
            return False
        
    def log_loss(self, y, prob):
        if y == prob:
            return 0.0
        print(f"prob={str(prob)}")
        print(f"y={str(y)}")
        a = math.log(prob)
        print(f"a={str(a)}")
        b = math.log(1-prob)
        print(f"b={str(b)}")
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
    
    def overall_log_loss(self):
        return (1/len(self._log_loss_list)* sum(self._log_loss_list))
