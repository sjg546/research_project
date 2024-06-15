import math
import numpy as np
import pandas as pd 
import decimal
class Player():
    def __init__(self,score:float):
        self._history: list[float] = [score]

class FiveThirtyEight():
    def __init__(self):
        self._initial_score = 1500
        self._player_map:dict[str:Player] = {}
        self._weight = 1
        self._sigma = 0.4
        self._v = 5
        self._curly = 250
        self._match_count = 0
        self._log_loss_list = []
        self._federer_rank = []
        self._test_year = {"right":0,"wrong":0}

    def predict_match(self, winner_id:str, loser_id:str, test_year: bool):
        self._match_count += 1
        if not winner_id in self._player_map:
            self._player_map[winner_id] = Player(1500.0)
        if not loser_id in self._player_map:
            self._player_map[loser_id] = Player(1500.0)

        winner: Player = self._player_map[winner_id]
        loser: Player = self._player_map[loser_id]

        if winner_id == "Roger Federer":
            self._federer_rank.append(self._player_map[winner_id]._history[-1])
        elif loser_id == "Roger Federer":
            self._federer_rank.append(self._player_map[loser_id]._history[-1])

        prob = self.pi_i_j(winner,loser)

        result = self._predict(prob)

        if test_year:
            if result:
               self._test_year["right"] += 1
            else:
               self._test_year["wrong"] += 1

        if prob > 0.5:
            self._log_loss_list.append(self.log_loss(1.0,prob))
        else:
            self._log_loss_list.append(self.log_loss(0.0, prob))

        self.update(self._player_map[winner_id], self._player_map[loser_id])

        return result
    
    def pi_i_j(self, winner:Player ,loser:Player):
        return math.pow((1+ math.pow(10,(loser._history[-1]-winner._history[-1])/400)) ,-1)
    
    def update(self,winner:Player,loser:Player):
        winner._history.append(winner._history[-1] + (self._kit(winner))*(self._weight - self.pi_i_j(winner, loser)))
        loser._history.append(loser._history[-1] + (self._kit(loser))*(-1*self.pi_i_j(loser, winner)))

    def _predict(self,probability):
        random_prob = np.random.uniform(0,1)
        if random_prob < probability:
            return True
        else:
            return False
                
    def _kit(self,player:Player):
        return self._curly/math.pow((len(player._history)+self._v),self._sigma)
    
    def log_loss(self, y, prob):
        if y == prob:
            return 0.0
        if y == 0.0 and prob == 1.0:
            return 0.0
        
        # print(f"prob={str(prob)}")
        # print(f"y={str(y)}")
        # a = math.log(prob)
        # print(f"a={str(a)}")
        # b = math.log(1-prob)
        # print(f"b={str(b)}")
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
    
    def overall_log_loss(self):
        return (1/len(self._log_loss_list)* sum(self._log_loss_list))
    
    def from_df(self, df,start_curly=250,stop_curly=251,start_v=5,stop_v=6,sigma_list=[0.4]):

        for i in range(start_curly,stop_curly,20):
            for j in range(start_v,stop_v,1):
                for k in sigma_list:
                    print(f"i={i}, j={j}, j={j}, k={k}")
                    self._curly = i
                    self._v = j
                    self._sigma = k
                    df['k_winner_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    df['k_loser_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    df['y_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    df['prob_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    df['loser_prob_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    df['log_loss_'+str(i)+"_" +str(j)+"_" +str(k)] = pd.Series(dtype='float')
                    #df['pred_class'] = pd.Series(dtype='float')

                    self._player_map.clear()
                    for index, row in df.iterrows():
                        winner_name = row['Winner']
                        loser_name = row['Loser']
                        if not winner_name in self._player_map:
                            self._player_map[winner_name] = Player(1500.0)
                        if not loser_name in self._player_map:
                            self._player_map[loser_name] = Player(1500.0)

                        winner: Player = self._player_map[winner_name]
                        loser: Player = self._player_map[loser_name]

                        prob = self.pi_i_j(winner,loser)
                        if(self._player_map[winner_name]._history[-1] > self._player_map[loser_name]._history[-1]):
                            df.loc[index,"y_"+str(i)+"_" +str(j)+"_" +str(k)] = 1.0
                            df.loc[index,"prob_"+str(i)+"_" +str(j)+"_" +str(k)] = prob
                            df.loc[index,"loser_prob_"+str(i)+"_" +str(j)+"_" +str(k)] = 1- prob

                            df.loc[index,"log_loss_"+str(i)+"_" +str(j)+"_" +str(k)] = self.log_loss(1.0,prob)
                            # df.loc[index,"loser_prob"] = 1-prob
                        else:
                            df.loc[index,"y_"+str(i)+"_" +str(j)+"_" +str(k)] = 0.0   
                            df.loc[index,"prob_"+str(i)+"_" +str(j)+"_" +str(k)] = prob
                            df.loc[index,"loser_prob_"+str(i)+"_" +str(j)+"_" +str(k)] = 1- prob
                            df.loc[index,"log_loss_"+str(i)+"_" +str(j)+"_" +str(k)] = self.log_loss(0.0,1-prob)

                            # df.loc[index,"loser_prob"] = 1-prob
                        
                        self.update(self._player_map[winner_name], self._player_map[loser_name])
                        # if index % 1000 == 0:
                        #     print(index)
                        df.loc[index,"k_winner_"+str(i)+"_" +str(j)+"_" +str(k)] = self._player_map[winner_name]._history[-1]
                        df.loc[index,"k_loser_"+str(i)+"_" +str(j)+"_" +str(k)] = self._player_map[loser_name]._history[-1]
        
        return df
