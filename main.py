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
from sklearn.metrics import log_loss
from surface_elo import SurfaceElo

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

# def log_loss(y, prob):
#         if y == prob:
#             return 0.0
#         # print(f"prob={str(prob)}")
#         # print(f"y={str(y)}")
#         # a = math.log(prob)
#         # print(f"a={str(a)}")
#         # b = math.log(1-prob)
#         # print(f"b={str(b)}")
#         return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))
    
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

# def merge_df(df1,df2):
def merge_datasets():
    frames = []
    for filename in os.listdir("odds_ds"):
        f = os.path.join("odds_ds", filename)
        # checking if it is a file
        # print(f)
        if os.path.isfile(f) and ".csv" in f:
            frames.append(bm_consensus.clean_data(f))

    for i in range(2001,2025):
        df1 = pd.read_csv("odds_ds/"+ str(i) +".csv")
        print(df1.dtypes)
        df1["Location"] = df1["Location"].astype(str)
        df1["WRank"] = pd.to_numeric(df1["WRank"])
        df1["LRank"] = pd.to_numeric(df1["LRank"])

        df1["Date"]=pd.to_datetime(df1['Date'])
        df1.loc[df1['Location'] == "Melbourne", 'Location'] = "Australian Open"
        # df1.loc[df1['Location'] == "Shanghai", 'Location'] = "Shanghai Masters"

        #df1.loc[df1['Location'] == "Vina del Mar", 'Location'] = "Santiago"
        df1.loc[df1['Tournament'] == "Bellsouth Open", 'Location'] = "Vina del Mar"

        df1.loc[df1['Location'] == "New York", 'Location'] = "US Open"
        df1.loc[df1['Location'] == "Cincinnati", 'Location'] = "Cincinnati Masters"
        df1.loc[df1['Tournament'] == "Masters Cup", "Location"] = "Tour Finals"
        df1.loc[df1['Tournament'] == "BNP Paribas Masters", "Location"] = "Paris Masters"
        df1.loc[df1['Tournament'] == "BNP Paribas", "Location"] = "Paris Masters"

        df1.loc[df1['Tournament'] == "Rogers Masters", "Location"] = "Canada Masters"
        df1.loc[df1['Tournament'] == "Wimbledon", "Location"] = "Wimbledon"
        df1.loc[df1['Tournament'] == "Topshelf Open", "Location"] = "s Hertogenbosch"
        df1.loc[df1['Tournament'] == "AEGON Championships", "Location"] = "Queen's Club"
        df1.loc[df1['Tournament'] == "French Open", "Location"] = "Roland Garros"
        df1.loc[df1['Tournament'] == "Internazionali BNL d'Italia", "Location"] = "Rome Masters"
        df1.loc[df1['Tournament'] == "Mutua Madrid Open", "Location"] = "Madrid Masters"
        df1.loc[df1['Tournament'] == "Monte Carlo Masters", "Location"] = "Monte Carlo Masters"
        df1.loc[df1['Tournament'] == "Sony Ericsson Open", "Location"] = "Miami Masters"
        df1.loc[df1['Tournament'] == "Ericsson Open", "Location"] = "Miami Masters"
        df1.loc[df1['Tournament'] == "NASDAQ-100 Open", "Location"] = "Miami Masters"

        df1.loc[df1['Tournament'] == "BNP Paribas Open", "Location"] = "Indian Wells Masters"
        df1.loc[df1['Tournament'] == "Dubai Tennis Championships", "Location"] = "Dubai"
        df1.loc[df1['Tournament'] == "Indian Wells TMS", "Location"] = "Indian Wells Masters"
        df1.loc[df1['Tournament'] == "Rome TMS", "Location"] = "Rome Masters"
        df1.loc[df1['Tournament'] == "Hamburg TMS", "Location"] = "Hamburg Masters"
        df1.loc[df1['Location'] == "Queens Club", "Location"] = "Queen's Club"
        df1.loc[df1['Tournament'] == "Montreal TMS", "Location"] = "Canada Masters"
        df1.loc[df1['Tournament'] == "Toronto TMS", "Location"] = "Canada Masters"
        df1.loc[df1['Tournament'] == "Madrid Masters", "Location"] = "Madrid Masters"

        #df1.loc[df1['Location'] == "Tour Finals", "Location"] = "Masters Cup"
        df1.loc[df1['Tournament'] == "Stuttgart TMS", "Location"] = "Stuttgart Masters"
        df1.loc[df1['Tournament'] == "Brasil Open", "Location"] = "Costa Do Sauipe"
        df1.loc[df1['Location'] == "'s-Hertogenbosch", "Location"] = "s Hertogenbosch"
        df1.loc[df1['Location'] == "St. Polten", "Location"] = "St. Poelten"
        df1.loc[df1['Tournament'] == "Pacific Life Open", "Location"] = "Indian Wells Masters"
        df1.loc[df1['Location'] == "Estoril ", "Location"] = "Estoril"
        df1.loc[df1['Location'] == "Rome", "Location"] = "Rome Masters"
        df1.loc[df1['Location'] == "Vienna ", "Location"] = "Vienna"
        df1.loc[df1['Location'] == "Dubai ", "Location"] = "Dubai"
        df1.loc[df1['Tournament'] == "Rogers Cup", "Location"] = "Canada Masters"
        df1.loc[df1['Tournament'] == "Vietnam Open", "Location"] = "Ho Chi Minh City"
        df1.loc[df1['Tournament'] == "Movistar Open", "Location"] = "Vina del Mar"
        df1.loc[df1['Location'] == "Portschach", "Location"] = "Poertschach"
        df1.loc[df1['Location'] == "Johannesburg ", "Location"] = "Johannesburg"
        df1.loc[df1['Tournament'] == "Mutua Madrileña Madrid Open", "Location"] = "Madrid Masters"
        df1.loc[df1['Tournament'] == "Shanghai Masters", 'Location'] = "Shanghai Masters"
        df1.loc[df1['Tournament'] == "VTR Open", 'Location'] = "Vina del Mar"


        # df1["WRank"] = df1["WRank"].dropna().astype(int)
        # df1["LRank"] = df1["WRank"].dropna().astype(int)

        # df1["Date"] = df1['Date'] - pd.Timedelta(1, unit='D')

        df1["Date"]=df1['Date'].dt.strftime("%Y%m%d").astype(int)

        df2 = pd.read_csv("tennis_atp/mens_atp/atp_matches_"+str(i)+".csv")
        print(df2.dtypes)
        df2["tourney_name"] = df2["tourney_name"].astype(str)
        df2["winner_rank"] = pd.to_numeric(df2["winner_rank"])
        df2["loser_rank"] = pd.to_numeric(df2["loser_rank"])

        # df2["winner_rank"] = df2["winner_rank"].dropna().astype(int)
        # df2["loser_rank"] = df2["loser_rank"].dropna().astype(int)

        new_df = pd.merge(df1, df2,  how='left', left_on=['Location','LRank','WRank'], right_on = ['tourney_name','loser_rank','winner_rank'])

        new_df.to_csv("joined/joined_"+str(i)+".csv")
        # df2 = 
        # df1["Date"].str.replace("/","").astype(int)
        # print(df1["Date"])
        # bm_consensus.clean_data("odds_ds/2001.csv")
        combined_dfs = pd.concat(frames)
def test_ll(y,prob):
    if y == prob:
        return 0.0
    if y == 0.0 and prob == 1.0:
        return 0.0
        
    return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))


def load_joined():
    frames = []
    for filename in os.listdir("joined"):
        f = os.path.join("joined", filename)
        print(f)
        frames.append(pd.read_csv(f))
    return pd.concat(frames)

def pi_i_j(winner_elo ,loser_elo):
    return math.pow((1+ math.pow(10,(loser_elo-winner_elo)/400)) ,-1)

start = 5
stop = 5
start_sigma = 0.4
start_curly = 100
start_v = 1
stop_sigma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
stop_curly = 300
stop_v = 10

# bm_consensus = BookmakersConsensus()
# # print(pd.to_datetime("31/12/2012",format='%Y%m%d'))
# # # print(combined_dfs)
# combined_dfs = load_joined()
# # # # bm_consensus.calculate_odds(combined_dfs)
# model = FiveThirtyEight()
# # surfaces = combined_dfs["Surface"].unique()
# # model = SurfaceElo(surfaces)
# c = combined_dfs.reset_index()
# a = model.from_df(c)
# d = []
# # for i in range(start,stop+1,5):
# #     d.append("y_" +str(i))
# #     d.append("prob_" +str(i))
# #     d.append("loser_prob_" +str(i))
# #     d.append("log_loss_" +str(i))
# #     d.append("k_winner_" +str(i))
# #     d.append("k_loser_" +str(i))
# for i in range(start_curly,stop_curly+1,20):
#     for j in range(start_v,stop_v+1,1):
#             for k in stop_sigma:
#                 d.append("y_" +str(i) +"_" +str(j)+"_" +str(k))
#                 d.append("prob_" +str(i)+"_" +str(j)+"_" +str(k))
#                 d.append("loser_prob_" +str(i)+"_" +str(j)+"_" +str(k))
#                 d.append("log_loss_" +str(i)+"_" +str(j)+"_" +str(k))
#                 d.append("k_winner_" +str(i)+"_" +str(j)+"_" +str(k))
#                 d.append("k_loser_" +str(i)+"_" +str(j)+"_" +str(k))
# d.append("y_250_5_0.4" )
# d.append("prob_250_5_0.4")
# d.append("loser_prob_250_5_0.4")
# d.append("log_loss_250_5_0.4")
# d.append("k_winner_250_5_0.4")
# d.append("k_loser_250_5_0.4") 
# d.append("k_prev_winner_250_5_0.4")
# d.append("k_prev_loser_250_5_0.4")

# e = ["Winner","Loser"] + d
# # print(e)
# b = a[e]
# b.to_csv("output_models/out_538_prev.csv")

# b.to_csv("output_models/out_538.csv")
# d = []
# for surface in surfaces:
#     d.append('surface_' + surface+ "_winner_prev")
#     d.append('surface_' + surface+ "_loser_prev")
#     d.append('surface_' + surface+ "_winner")
#     d.append('surface_' + surface+ "_loser")
# e = ["Winner","Loser"] + d + ["prob_winner", "prob_loser"]
# b = a[e]
# b.to_csv("output_models/out_surface_elo_prev.csv")
a = pd.read_csv("output_models/out_538_prev.csv",usecols=["Winner","Loser","y_250_5_0.4","k_prev_winner_250_5_0.4","k_prev_loser_250_5_0.4","prob_250_5_0.4","loser_prob_250_5_0.4","log_loss_250_5_0.4"])
# a.to_csv("output_models/out_538_optimal.csv")

c = pd.read_csv("output_models/out_surface_elo_prev.csv",usecols=["prob_winner","prob_loser","surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"])

joined = a.join(c)

sigma = 0.85

surface_fields = ["surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"]


joined["combined_prob"] = sigma*joined["prob_250_5_0.4"] + (1-sigma)*joined["prob_winner"]
for index, row in joined.iterrows():   
    for surface in surface_fields:
        if not pd.isna(row[surface]) and "winner" in surface:
            joined.loc[index,"combined_winner_elo"] = sigma*row["k_prev_winner_250_5_0.4"] + (1-sigma)*row[surface]
        if not pd.isna(row[surface]) and "loser" in surface:    
            joined.loc[index,"combined_loser_elo"] = sigma*row["k_prev_loser_250_5_0.4"] + (1-sigma)*row[surface]
        
joined["combined_elo_prob"] = joined.apply(lambda row: pi_i_j(row["combined_winner_elo"],row["combined_loser_elo"]),axis=1)
# print(joined["combined_elo_prob"])
# joined.to_csv("output_models/combined_surface.csv")
# joined = pd.read_csv("output_models/combined_surface.csv")
# print(joined["prob_winner"])
sum_gr_0_5_raw = len(joined[(joined["prob_250_5_0.4"]>0.5)])
sum_gr_0_5 = len(joined[(joined["combined_prob"]>0.5)])
sum_gr_0_5_comb = len(joined[(joined["combined_elo_prob"]>0.5)])

print(f"accuracy comb prob = {sum_gr_0_5/len(joined)}")
print(f"accuracy raw = {sum_gr_0_5_raw/len(joined)}")
print(f"accuracy comb elo= {sum_gr_0_5_comb/len(joined)}")

for index, row in joined.iterrows():    
   if(row["combined_prob"] >0.5 ):
       joined.loc[index,"combined_log_loss"] = test_ll(1.0,row["combined_prob"])
       # df.loc[index,"loser_prob"] = 1-prob
   else:
       joined.loc[index,"combined_log_loss"] = test_ll(0.0,1.0-row["combined_prob"])

for index, row in joined.iterrows():    
   if(row["combined_winner_elo"] > row["combined_loser_elo"]):
       joined.loc[index,"combined_log_loss_elo"] = test_ll(1.0,row["combined_elo_prob"])
       # df.loc[index,"loser_prob"] = 1-prob
   else:
       joined.loc[index,"combined_log_loss_elo"] = test_ll(0.0,1.0-row["combined_elo_prob"])

print(f"logloss comb prob = {np.mean(joined['combined_log_loss'])}")
print(f"logloss = {np.mean(joined['log_loss_250_5_0.4'])}")
print(f"logloss comb elo= {np.mean(joined['combined_log_loss_elo'])}")

# player = 'Federer R.'

# player_plot = a.query(f"Winner == '{player}' or Loser == '{player}'")
# player_plot = player_plot.reset_index()
# player_plot["elo"] = player_plot[["Winner","Loser","k_winner_260_5_0.4","k_loser_260_5_0.4"]].apply(lambda x : x["k_winner_260_5_0.4"] if(x["Winner"] == player) else x["k_loser_260_5_0.4"], axis=1)
# print(player_plot[["Winner","Loser","elo"]])
# player_plot["elo"].plot()
# plt.show()
# print(c)

# # 0.0,0.4640522024570126,0.5359477975429874
# # print(log_loss([0],[0.4640522024570126], labels=[0,]))
# for i in range(start,stop+1,5):
    
#     sum_gr_0_5 = len(c[(c['prob_'+str(i)]>0.5)])
#     print(f"k = {i} accuracy = {sum_gr_0_5/len(c)}")
#     # logloss = log_loss(c['y'],c['prob'], labels=[1.0,0.0])
#     print(f"k = {i} logloss = {np.mean(c['log_loss_'+str(i)])}")

# for i in range(start_curly,stop_curly+1,20):
#     for j in range(start_v,stop_v+1,1):
#         for k in stop_sigma:
#             sum_gr_0_5 = len(c[(c['prob_'+str(i)+"_" +str(j)+"_" +str(k)]>0.5)])
#             print(f"curly = {i}, v = {j}, sigma = {k} accuracy = {sum_gr_0_5/len(c)}")
#             # logloss = log_loss(c['y'],c['prob'], labels=[1.0,0.0])
#             print(f"curly = {i}, v = {j}, sigma = {k} logloss = {np.mean(c['log_loss_'+str(i)+'_' +str(j)+'_' +str(k)])}")

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

            


