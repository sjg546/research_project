import numpy as np
import math
class LogisticModel():
    def __init__(self):
        self._intercept = 0.5
        # self._slope = 0.02
        self._slope = 0.000565
        self._circle = 0.0004196
        self._threshold = 0.5
        self._log_loss_list = []
        self._multiple_accuracy = []
        self.multiple_log_loss = []

    def predict_match(self, winner_points, loser_points):
        diff_points = winner_points - loser_points
        probability = self._calculate_probability_log(diff_points)
        return_val = 1
        if loser_points < winner_points:
            if probability > self._threshold:
                self._log_loss_list.append(self.log_loss(1.0,probability))
            else:
                self._log_loss_list.append(self.log_loss(0.0, probability))

        # if loser_points > winner_points:
        #     return 1
        return self._predict(probability)
    
    def _calculate_probability(self,diff_points):
        probability = self._intercept + (self._slope *diff_points)
        return probability
    
    def _predict(self,probability):
        # print(probability)
        # random_prob = np.random.uniform(0,1)
        if probability > self._threshold :
            return 0
        else:
            return 1
        
    def _calculate_probability_log(self, diff_points):
        return 1/(1+math.exp(-1*(self._slope*diff_points)))
    
    def log_loss(self, y, prob):
        if y == prob:
            return 0.0
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
    
    def overall_log_loss(self):
        a = (1/len(self._log_loss_list)* sum(self._log_loss_list))
        self._log_loss_list.clear()
        return a