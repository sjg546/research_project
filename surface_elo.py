import math
import numpy as np
import pandas as pd 
import decimal
class Player():
    def __init__(self,score:float):
        self._history: list[float] = [score]
        self.surface_elo = {}

class SurfaceElo():
    def __init__(self, surface_types):
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
        self.surface_types = surface_types

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
    
    def pi_i_j(self, winner:Player ,loser:Player,surface:str):
        return math.pow((1+ math.pow(10,(loser.surface_elo[surface][-1]-winner.surface_elo[surface][-1])/400)) ,-1)
    
    def update(self,winner:Player, loser:Player ,surface:str, won: bool):
        winner.surface_elo[surface].append(winner.surface_elo[surface][-1] + (25)*(self._weight - self.pi_i_j(winner, loser,surface)))
        loser.surface_elo[surface].append(loser.surface_elo[surface][-1] - (25)*(self._weight - self.pi_i_j(loser, winner,surface)))

    def _predict(self,probability):
        random_prob = np.random.uniform(0,1)
        if random_prob < probability:
            return True
        else:
            return False
                
    def _kit(self,player:Player,surface_type:str):
        return self._curly/math.pow((len(player.surface_elo[surface_type])+self._v),self._sigma)

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
    
    def from_df(self, df):
        for surface in self.surface_types:
            df['surface_' + surface+ "_winner"] = pd.Series(dtype='float')
            df['surface_' + surface+ "_loser"] = pd.Series(dtype='float')
            df['prob_winner'] = pd.Series(dtype='float')
            df['prob_loser'] = pd.Series(dtype='float')

        self._player_map.clear()
        for index, row in df.iterrows():
            winner_name = row['Winner']
            loser_name = row['Loser']
            if not winner_name in self._player_map:
                self._player_map[winner_name] = Player(1500.0)
                for surface in self.surface_types:
                    self._player_map[winner_name].surface_elo[surface] = [1500]
            if not loser_name in self._player_map:
                self._player_map[loser_name] = Player(1500.0)
                self._player_map[loser_name].surface_elo[surface] = [1500]
                for surface in self.surface_types:
                    self._player_map[loser_name].surface_elo[surface] = [1500]

            winner: Player = self._player_map[winner_name]
            loser: Player = self._player_map[loser_name]

            surface = row["Surface"]
            prob = self.pi_i_j(winner,loser,surface)

            df.loc[index,'surface_' + surface + "_winner_prev"] = winner.surface_elo[surface][-1]
            df.loc[index,'surface_' + surface + "_loser_prev"] = loser.surface_elo[surface][-1]  

            self.update(winner,loser,surface, True)

            df.loc[index,'surface_' + surface + "_winner"] = winner.surface_elo[surface][-1]
            df.loc[index,'surface_' + surface + "_loser"] = loser.surface_elo[surface][-1]      

            df.loc[index,'prob_winner'] = prob
            df.loc[index,'prob_loser'] = 1 - prob        
        
        return df
