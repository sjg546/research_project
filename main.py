import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from logisticmodel import LogisticModel
from elommr import ELOMMR
import json
import numpy as np
import math
from resultsparser import ResultsParser
from tests import Tests
from geopy.geocoders import Nominatim 
from linearmodel import LinearModel
from kfactor import KFactor
from fivethirtyeight import FiveThirtyEight
from bookmakers_consensus import BookmakersConsensus
directory = "tennis_atp/mens_atp"

# parser = ResultsParser()
# parser.read_ratings()
elo_results = {"right":0,"wrong":0}
rank_results = {"right":0, "wrong":0}
logistic_result = {"right":0,"wrong":0}
kfactor_result = {"right":0,"wrong":0}
five_thirty_eight_result = {"right":0,"wrong":0}
_log_loss_list = []
# geolocator = Nominatim(user_agent = "geoapiExercises")
# location = geolocator.geocode("Nottingham")
# print("Country Name: ", location)
hand_bonus_base = 100
height_bonus_base = 10

federer_k_list = []
federer_538_list = []

def pi_i_j( winner ,loser):
    return math.pow((1+ math.pow(10,(loser-winner)/400)) ,-1)

def log_loss(y, prob):
        if y == prob:
            return 0.0
        # print(f"prob={str(prob)}")
        # print(f"y={str(y)}")
        # a = math.log(prob)
        # print(f"a={str(a)}")
        # b = math.log(1-prob)
        # print(f"b={str(b)}")
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
    
def overall_log_loss():
        return (1/len(_log_loss_list)* sum(_log_loss_list))

def predict_matches():
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            # print(f)
            df = pd.read_csv(f)
            for ind in df.index:
                contest_date=df["tourney_id"][ind]
                winner_name = df["winner_name"][ind]
                winner_rank = df["winner_rank"][ind]
                winner_hand = df["winner_hand"][ind]
                winner_height = df["winner_ht"][ind]
                loser_name = df["loser_name"][ind]
                loser_rank = df["loser_rank"][ind]
                loser_hand = df["loser_hand"][ind]
                loser_height = df["loser_ht"][ind]

                ratings = parser.get_current_rankings(str(contest_date))

                if winner_name in ratings and loser_name in ratings:
                    base_winner_rating = ratings[winner_name]
                    base_loser_rating = ratings[loser_name]
                    winner_rating = base_winner_rating
                    loser_rating = base_loser_rating
                    prob = pi_i_j(winner_rating,loser_rating)

                    # if abs(winner_rating-loser_rating) < 100:
                    #     if winner_hand == "R" and loser_hand == "L":
                    #         loser_rating += base_loser_rating * 0.01 # Best players are right handed?, look into scaling up left handed rating if the rating difference is less?
                    #     elif loser_hand == "R" and winner_hand == "L":
                    #         winner_rating += base_winner_rating * 0.01 # Best players are right handed?, look into scaling up left handed rating if the rating difference is less?

                    #     if not pd.isna(winner_height) and not pd.isna(loser_height):
                    #         if  winner_height > loser_height:
                    #             winner_rating += base_winner_rating * 0.02
                    #         elif loser_height > winner_height:
                    #             loser_rating += base_loser_rating * 0.02                     

                    if prob > 0.5:
                        _log_loss_list.append(log_loss(1.0,prob))
                        elo_results["right"] += 1
                    else:
                        _log_loss_list.append(log_loss(0.0,prob))
                        elo_results["wrong"] += 1
                
                else:
                    if not pd.isna(winner_rank) and not pd.isna(loser_rank):
                        if winner_rank < loser_rank:
                            elo_results["right"] += 1
                        else:
                            elo_results["wrong"] += 1

                if not pd.isna(winner_rank) and not pd.isna(loser_rank):
                    if winner_rank < loser_rank:
                        rank_results["right"] += 1
                    else:
                        rank_results["wrong"] += 1

    print("-------ELO RESULTS-------")
    print(elo_results)
    pct = elo_results["right"]/ (elo_results["right"] + elo_results["wrong"])
    print(pct)
    print("-------------------------")
    print(overall_log_loss())
    print("-----Ratings Results-----")
    print(rank_results)
    pct = rank_results["right"]/ (rank_results["right"] + rank_results["wrong"])
    print(pct)
    print("-------------------------")


def generate_rankings(directory):
    player_names_to_id = {}
    print(directory)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            # print(f)
            df = pd.read_csv(f)
            previous_id = ""
            player_performance = {}

            for ind in df.index: 
                if df["round"][ind] != "RR":
                    current_tourney_id = df["tourney_id"][ind]
                    if current_tourney_id != previous_id:
                        print("new_tourney")
                        b =dict(reversed(sorted(player_performance.items(), key=lambda item: item[1])))

                        player_list = []
                        standings = []                                                
                        if player_performance:
                            a = max(player_performance.values())
                            current_max = a
                            current_rating = 1
                            print(a)

                            for entry in b:
                                player_list.insert(0, entry)
                                if player_performance[entry] == current_max:
                                    ranking = current_rating
                                else:
                                    current_max = player_performance[entry]
                                    current_rating += 1
                                    ranking = current_rating
                                standings.append([player_names_to_id[entry],ranking])

                            outjson = {"id":str(df["tourney_id"][ind]),  "date":str(df["tourney_date"][ind]),"standings":standings}
                            filename = str(df["tourney_id"][ind]) +".json"
                            with open("test/contests/"+filename, "w") as outfile: 
                                json.dump(outjson,outfile)


                            # elommr.calculate_round(player_list)                    
                            player_performance.clear()

                    previous_id = current_tourney_id
                    winner_id = int(df["winner_id"][ind])
                    loser_id = int(df["loser_id"][ind])

                    if not winner_id in player_names_to_id:
                        player_names_to_id[winner_id] = df["winner_name"][ind] 
                    if not loser_id in player_names_to_id:
                        player_names_to_id[loser_id] = df["loser_name"][ind] 

                    if not winner_id in player_performance:
                        player_performance[winner_id] = 1
                    else:
                        player_performance[winner_id] += 1
                    if not loser_id in player_performance:
                        player_performance[loser_id] = 0             

def predict_logistic_rankings():
    model = LogisticModel()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            # print(f)
            df = pd.read_csv(f)
            for ind in df.index:
                winner_rank_points = df["winner_rank_points"][ind]
                loser_rank_points = df["loser_rank_points"][ind]
                if not pd.isna(winner_rank_points) and not pd.isna(loser_rank_points): 
                    prediction = model.predict_match(winner_rank_points,loser_rank_points)       
                    if prediction == 0 :
                        logistic_result["right"] += 1
                    elif prediction == 1:
                        logistic_result["wrong"] += 1

    print(model.overall_log_loss())
    print("-------Logistic RESULTS-------")
    print(logistic_result)
    pct = logistic_result["right"]/ (logistic_result["right"] + logistic_result["wrong"])
    print(pct)
    print("-------------------------")

def predict_kfactor_rankings():
    model = KFactor()
    starting_k = 5.0
    ending_k = 10.0
    current_k = 1.0
    # while current_k <= ending_k:
    model._k = 30.0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            test_year = False
            if "2019" in f:
                test_year = True

            # print(f)
            df = pd.read_csv(f)
            for ind in df.index:
                winner_name = df["winner_name"][ind]
                loser_name = df["loser_name"][ind]
                if not pd.isna(winner_name) and not pd.isna(loser_name):                    
                    if model.predict_match(winner_name,loser_name,test_year):
                        kfactor_result["right"] += 1
                    else:
                        kfactor_result["wrong"] += 1
    accuracy = kfactor_result["right"]/ (kfactor_result["right"] + kfactor_result["wrong"])
    model.multiple_log_loss.append(model.overall_log_loss())
    model._multiple_accuracy.append(accuracy)
    kfactor_result.clear
    current_k += 5.0
    print(model.multiple_log_loss)
    print(model._multiple_accuracy)

    print(model._test_year)
    pct = model._test_year["right"]/ (model._test_year["right"] + model._test_year["wrong"])
    print(pct)

    global federer_k_list
    federer_k_list = model._federer_rank

    # xs = [x for x in range(len(model._federer_rank))]

    # plt.plot(xs, model._federer_rank)
    # plt.show()
    # # Make sure to close the plt object once done
    # plt.close()

    # print(model.overall_log_loss())
    # # print(model._player_map["Roger Federer"]._history)
    # print("-------KFactor RESULTS-------")
    # print(kfactor_result)
    # pct = kfactor_result["right"]/ (kfactor_result["right"] + kfactor_result["wrong"])
    # print(pct)
    # print("-------------------------")

def predict_538_rankings():
    model = FiveThirtyEight()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            test_year = False
            if "2019" in f:
                test_year = True
            # print(f)
            df = pd.read_csv(f)
            for ind in df.index:
                winner_name = df["winner_name"][ind]
                loser_name = df["loser_name"][ind]
                if not pd.isna(winner_name) and not pd.isna(loser_name):                    
                    if model.predict_match(winner_name,loser_name,test_year):
                        five_thirty_eight_result["right"] += 1
                    else:
                        five_thirty_eight_result["wrong"] += 1

    print(model.overall_log_loss())
    # print(model._player_map["Roger Federer"]._history)
    print("-------538 RESULTS-------")
    print(five_thirty_eight_result)
    pct = five_thirty_eight_result["right"]/ (five_thirty_eight_result["right"] + five_thirty_eight_result["wrong"])
    print(pct)
    print("-------------------------")
    print(model._test_year)
    pct = model._test_year["right"]/ (model._test_year["right"] + model._test_year["wrong"])
    print(pct)

    global federer_538_list
    federer_538_list = model._federer_rank
    # plt.plot(xs, model._federer_rank)
    # plt.show()
    # # Make sure to close the plt object once done
    # plt.close()


bm_consensus = BookmakersConsensus()
df = bm_consensus.clean_data("odds_ds/2001.csv")
bm_consensus.calculate_odds(df)
# predict_matches()
# predict_logistic_rankings()
# predict_kfactor_rankings()
# predict_538_rankings()
# xs = [x for x in range(len(federer_k_list))]

# plt.plot(xs, federer_k_list, label = "K Factor")
# plt.plot(xs, federer_538_list, label = "Five Thirty Eight")
# plt.legend()
# plt.show()

# t = Tests()
# t.handed_test("tennis_atp/mens_atp")
# t.height_test("tennis_atp/mens_atp")
# player_names_to_id = {}

# elommr = ELOMMR()

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         print(f)
#         # print(f)
#         df = pd.read_csv(f)
#         previous_id = ""
#         player_performance:dict = {}

#         for ind in df.index:
#             current_tourney_id = df["tourney_id"][ind]
#             if current_tourney_id != previous_id:
#                 print("new_tourney")
#                 b =dict(reversed(sorted(player_performance.items(), key=lambda item: item[1])))

#                 player_list = []
#                 standings = []
#                 if player_performance:
#                     a = max(player_performance.values())
#                     print(a)

#                     for entry in b:
#                         player_list.insert(0, entry)
#                         ranking = int(a+1 -player_performance[entry])
#                         standings.append([player_names_to_id[entry],ranking])

#                     outjson = {"date":str(df["tourney_date"][ind]),"standings":standings}
#                     print(outjson)
#                     filename = str(df["tourney_date"][ind]) +".json"
#                     with open("test/contests/"+filename, "w") as outfile: 
#                         json.dump(outjson,outfile)


#                     # elommr.calculate_round(player_list)                    
#                     player_performance.clear()

#             previous_id = current_tourney_id
#             winner_id = int(df["winner_id"][ind])
#             loser_id = int(df["loser_id"][ind])

#             if not winner_id in player_names_to_id:
#                 player_names_to_id[winner_id] = df["winner_name"][ind] 
#             if not loser_id in player_names_to_id:
#                 player_names_to_id[loser_id] = df["loser_name"][ind] 
 
#             if not winner_id in player_performance:
#                 player_performance[winner_id] = 1
#             else:
#                 player_performance[winner_id] += 1
#             if not loser_id in player_performance:
#                 player_performance[loser_id] = 0             
 

#             # Create a map for player id's to their names, no we only use ids going forward

#             # player_list = [winner_id,loser_id]
#             # elommr.calculate_round(player_list)

    

# for player in elommr.player_list:
#     print(str(player_names_to_id[elommr.player_list[player].player_id]) + ":"+ str(elommr.player_list[player].mean))

# # print(player_names_to_id)

            


