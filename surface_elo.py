import math
import numpy as np
import pandas as pd 
import decimal
class Player():
    def __init__(self,score:float):
        self._history: list[float] = [score]
        self.surface_elo = {}

class SurfaceElo():
    def __init__(self, surface_types,curly=250,v=5,sigma=0.4):
        self._initial_score = 1500
        self._player_map:dict[str:Player] = {}
        self._weight = 1
        self._sigma = sigma
        self._v = v
        self._curly = curly
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
    def pi_i_j_2(self,winner ,loser):
        return math.pow((1+ math.pow(10,(loser-winner)/400)) ,-1)

    def test_ll(self,y,prob):
        if y == prob:
            return 0.0
        if y == 0.0 and prob == 1.0:
            return 0.0
            
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
   
    def run_metrics(self,df:pd.DataFrame,sigma,prob):
        five_thirty_eight_output = pd.read_csv("output_models/temp_538.csv", usecols=["Winner","Loser",
                                                                                          f"y_{self._curly}_{self._v}_{self._sigma}",f"k_prev_winner_{self._curly}_{self._v}_{self._sigma}",
                                                                                          f"k_winner_{self._curly}_{self._v}_{self._sigma}",f"k_loser_{self._curly}_{self._v}_{self._sigma}",
                                                                                          f"k_prev_loser_{self._curly}_{self._v}_{self._sigma}",f"prob_{self._curly}_{self._v}_{self._sigma}",
                                                                                          f"loser_prob_{self._curly}_{self._v}_{self._sigma}",f"log_loss_{self._curly}_{self._v}_{self._sigma}"])

        worked_df = df[["prob_winner","prob_loser","surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"]]
        joined = five_thirty_eight_output.join(worked_df)

        surface_fields = ["surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"]
        
        if not prob:
            for index, row in joined.iterrows():   
                for surface in surface_fields:
                    if not pd.isna(row[surface]) and "winner" in surface:
                        joined.loc[index,"combined_winner_elo"] = sigma*row[f"k_prev_winner_{self._curly}_{self._v}_{self._sigma}"] + (1-sigma)*row[surface]
                        joined.loc[index,"combined_future_winner_elo"] = sigma*row[f"k_winner_{self._curly}_{self._v}_{self._sigma}"] + (1-sigma)*row[surface]

                    if not pd.isna(row[surface]) and "loser" in surface:    
                        joined.loc[index,"combined_loser_elo"] = sigma*row[f"k_prev_loser_{self._curly}_{self._v}_{self._sigma}"] + (1-sigma)*row[surface]
                        joined.loc[index,"combined_future_loser_elo"] = sigma*row[f"k_loser_{self._curly}_{self._v}_{self._sigma}"] + (1-sigma)*row[surface]

            joined["combined_prob"] = joined.apply(lambda row: self.pi_i_j_2(row["combined_winner_elo"],row["combined_loser_elo"]),axis=1)
            joined["sum_elo"] = joined["combined_winner_elo"] - joined["combined_loser_elo"]
            # joined = joined.drop(joined[joined.sum_elo == 0.0].index)
            # joined = joined.reset_index()

        else:
            joined["combined_prob"] = sigma*joined[f"prob_{self._curly}_{self._v}_{self._sigma}"] + (1-sigma)*joined["prob_winner"]

        for index, row in joined.iterrows():    
            if(row["combined_prob"] >0.5 ):
                joined.loc[index,"combined_log_loss"] = self.test_ll(1.0,row["combined_prob"])
                # df.loc[index,"loser_prob"] = 1-prob
            else:
                joined.loc[index,"combined_log_loss"] = self.test_ll(0.0,1.0-row["combined_prob"])

        if(row["combined_prob"] < 0.5 ):
            joined.loc[index,"predicted"] = 0.0
        else:
            joined.loc[index,"predicted"] = 1.0

        correct_predictions = len(joined[joined["combined_prob"] > 0.5])
        total_predictions = len(joined["combined_prob"])
        total_prob = np.sum(joined["combined_prob"])
        print(f"Surface Elo Prob, Delta={self._curly}, Nu={self._v}, Sigma={self._sigma}, Calibration = {total_prob/correct_predictions}")
        print(f"Surface Elo Prob, Delta={self._curly}, Nu={self._v}, Sigma={self._sigma}, Accuracy = {correct_predictions/total_predictions}")
        print(f"Surface Elo Prob, Delta={self._curly}, Nu={self._v}, Sigma={self._sigma}, Logloss = {np.mean(joined['combined_log_loss'])}")
        
        return joined