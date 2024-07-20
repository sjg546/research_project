import numpy as np
import math
import pandas as pd
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
                self._log_loss_list.append(self.log_loss(0.0, 1- probability))

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

    def from_df(self, df):
        df['prob'] = pd.Series(dtype='float')
        df['loser_prob'] = pd.Series(dtype='float')
        df['log_loss'] = pd.Series(dtype='float')
        #df['pred_class'] = pd.Series(dtype='float')

        for index, row in df.iterrows():
            winner_points = row['winner_rank_points']
            loser_points = row['loser_rank_points']
            diff_points = winner_points - loser_points
            probability = self._calculate_probability_log(diff_points)

            if(probability > 0.5):
                df.loc[index,"y"] = 1.0
                df.loc[index,"prob"] = probability
                df.loc[index,"loser_prob"] = 1- probability

                df.loc[index,"log_loss"] = self.log_loss(1.0,probability)
                # df.loc[index,"loser_prob"] = 1-prob
            else:
                df.loc[index,"y"] = 0.0   
                df.loc[index,"prob"] = probability
                df.loc[index,"loser_prob"] = 1- probability
                df.loc[index,"log_loss"] = self.log_loss(0.0,1-probability)
                    
        return df
    
    def run_metrics(self,df:pd.DataFrame):
        correct_predictions = len(df[df["prob"] > 0.5])
        total_predictions = len(df["prob"])
        total_prob = np.sum(df["prob"])
        print(f"Logistic Model, Calibration = {total_prob/correct_predictions}")
        print(f"Logistic Model, Accuracy = {correct_predictions/total_predictions}")
        print(f"Surface Elo Prob, Logloss = {np.mean(df['log_loss'])}")
