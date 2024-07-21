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
from sklearn.calibration import calibration_curve
from surface_elo import SurfaceElo
from sklearn.metrics import confusion_matrix

'''
1 = Logistic
2 = K Factor
3 = Five Thirty Eight
4 = Five Thirty Eight - Surface ELO
5 = Five Thirty Eight - Surface PROB
6 = Bookmakers Consensus
'''
MODEL_TO_USE=4

'''
True = Build model and save to file
False = Use presaved model
'''
BUILD_MODEL=False

K_VALUE = 30

FIVE_THIRTY_EIGHT_DELTA = 240
FIVE_THIRTY_EIGHT_SIGMA = 0.4
FIVE_THIRTY_EIGHT_NU = 5

SURFACE_SIGMA = 0.85

directory = "tennis_atp/mens_atp"

_log_loss_list = []

def pi_i_j( winner ,loser):
    return math.pow((1+ math.pow(10,(loser-winner)/400)) ,-1)
    
def overall_log_loss():
        return (1/len(_log_loss_list)* sum(_log_loss_list))


def calbration_calc(df:pd.DataFrame,rank_1_name,rank2_name,prob_field_name):
    base_x = []
    base_y = []
    actual_x = []
    actual_y = []

    dif_mapping = {}
    for index, row in df.iterrows():
        rank_diff = row[rank_1_name]-row[rank2_name]
        row_prob = row[prob_field_name]
        base_x.append(rank_diff)
        base_y.append(row_prob)
    #     if rank_diff >= 0:
    #         if rank_diff in dif_mapping:
    #             dif_mapping[rank_diff].append(1.0)
    #         else:
    #             dif_mapping[rank_diff] = [1.0]
    #     elif rank_diff < 0:
    #         if rank_diff in dif_mapping:
    #             dif_mapping[rank_diff].append(0.0)
    #         else:
    #             dif_mapping[rank_diff] = [0.0]
    # for entry in dif_mapping:
    #     actual_x.append(entry)
    #     a = 0.0
    #     for i in dif_mapping[entry]:
    #         a= a + i

    #     prob = a/float(len(dif_mapping[entry]))
    #     actual_y.append(prob)



    plt.plot(base_x, base_y, label = "Accuracy",linestyle='None',marker='.')
    # plt.plot(actual_x, actual_y, label = "Accuracy",linestyle='None',marker='x')

    plt.show()

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
        # print(f)
        frames.append(pd.read_csv(f))
    return pd.concat(frames)

def pi_i_j(winner_elo ,loser_elo):
    return math.pow((1+ math.pow(10,(loser_elo-winner_elo)/400)) ,-1)

def build_five_thirty_eight(model,combined_dfs):
    df = model.from_df(combined_dfs,FIVE_THIRTY_EIGHT_DELTA,FIVE_THIRTY_EIGHT_DELTA+1,
                       FIVE_THIRTY_EIGHT_NU,FIVE_THIRTY_EIGHT_NU+1,
                       [FIVE_THIRTY_EIGHT_SIGMA])
    
    column_list = []
    column_list.append(f"y_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"prob_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"loser_prob_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"log_loss_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"k_winner_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"k_loser_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"k_prev_winner_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")
    column_list.append(f"k_prev_loser_{FIVE_THIRTY_EIGHT_DELTA}_{FIVE_THIRTY_EIGHT_NU}_{FIVE_THIRTY_EIGHT_SIGMA}")

    final_columns = ["Winner","Loser"] + column_list
    df_to_save = df[final_columns]
    df_to_save.to_csv("output_models/temp_538.csv")

    return df_to_save

def build_k_factor(model,combined_dfs):
    df = model.from_df(combined_dfs,K_VALUE,K_VALUE+1)
    column_list = []
    column_list.append(f"y_{K_VALUE}")
    column_list.append(f"prob_{K_VALUE}")
    column_list.append(f"loser_prob_{K_VALUE}")
    column_list.append(f"log_loss_{K_VALUE}")
    column_list.append(f"k_winner_{K_VALUE}")
    column_list.append(f"k_loser_{K_VALUE}")
    final_columns = ["Winner","Loser"] + column_list
    df_to_save = df[final_columns]
    df_to_save.to_csv("output_models/temp_k_factor.csv")
    return df_to_save

def build_surface_elo(model,combined_dfs):
    surfaces = combined_dfs["Surface"].unique()
    df = model.from_df(combined_dfs)
    column_list = []
    for surface in surfaces:
        column_list.append('surface_' + surface+ "_winner_prev")
        column_list.append('surface_' + surface+ "_loser_prev")
        column_list.append('surface_' + surface+ "_winner")
        column_list.append('surface_' + surface+ "_loser")

    final_columns = ["Winner","Loser"] + column_list + ["prob_winner", "prob_loser"]
    df_to_save = df[final_columns]
    df_to_save.to_csv("output_models/temp_surface.csv")
    return df_to_save

def build_logistics(model,combined_dfs):
    df = model.from_df(combined_dfs)
    column_list = ["prob","log_loss"]
    final_columns = ["Winner","Loser"] + column_list
    df_to_save = df[final_columns]
    df_to_save.to_csv("output_models/temp_log.csv")

if BUILD_MODEL:
    combined_dfs = load_joined()
    combined_dfs = combined_dfs.reset_index()

if MODEL_TO_USE == 1:
    model = LogisticModel()
    if BUILD_MODEL:
        df = build_logistics(model,combined_dfs)
    else:
        df = pd.read_csv("output_models/temp_log.csv")

    print(df)
    model.run_metrics(df)

elif MODEL_TO_USE == 2:
    model = KFactor()
    if BUILD_MODEL:
        df = build_k_factor(model)
    else:
        df = pd.read_csv("output_models/temp_k_factor.csv")

    model.run_metrics(df, K_VALUE)

elif MODEL_TO_USE == 3:
    model = FiveThirtyEight()
    if BUILD_MODEL:
        df = build_five_thirty_eight(model)
    else:
        df = pd.read_csv("output_models/temp_538.csv")

    model.run_metrics(df, FIVE_THIRTY_EIGHT_DELTA,FIVE_THIRTY_EIGHT_NU,FIVE_THIRTY_EIGHT_SIGMA)
elif MODEL_TO_USE == 4:
    combined_dfs = load_joined()
    combined_dfs = combined_dfs.reset_index()

    surfaces = combined_dfs["Surface"].unique()
    model = SurfaceElo(surfaces,FIVE_THIRTY_EIGHT_DELTA,FIVE_THIRTY_EIGHT_NU,FIVE_THIRTY_EIGHT_SIGMA)
    if BUILD_MODEL:
        df = build_surface_elo(model,combined_dfs)
    else:
        df = pd.read_csv("output_models/temp_surface.csv")

    df = model.run_metrics(df,SURFACE_SIGMA,False)

elif MODEL_TO_USE == 5:
    combined_dfs = load_joined()
    combined_dfs = combined_dfs.reset_index()

    surfaces = combined_dfs["Surface"].unique()
    model = SurfaceElo(surfaces,FIVE_THIRTY_EIGHT_DELTA,FIVE_THIRTY_EIGHT_NU,FIVE_THIRTY_EIGHT_SIGMA)
    if BUILD_MODEL:
        df = build_surface_elo(model,combined_dfs)
    else:
        df = pd.read_csv("output_models/temp_surface.csv")

    df = model.run_metrics(df,SURFACE_SIGMA,True)
elif MODEL_TO_USE == 6:
    model = BookmakersConsensus()  
    if BUILD_MODEL:
        model.calculate_odds(combined_dfs)
        combined_dfs.to_csv("output_models/temp_bookmakers.csv")
    df = pd.read_csv("output_models/temp_bookmakers.csv")

    df = model.run_metrics(combined_dfs)
else:
    print("Error no valid model selected")
    exit(1)


player = 'Federer R.'
# ELO PLOTS
# player_plot = df.query(f"Winner == '{player}' or Loser == '{player}'")
# player_plot = player_plot.reset_index()
# b = pd.read_csv("output_models/temp_k_factor.csv")
# player_plot2 = b.query(f"Winner == '{player}' or Loser == '{player}'")
# player_plot2 = player_plot2.reset_index()

# a = player_plot.iloc[300:330]
# b = player_plot2.iloc[300:330]
# c = a.merge(b)
# print(c[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo","k_winner_240_5_0.4","k_loser_240_5_0.4","k_winner_30","k_loser_30"]])
# a["elo"] = a[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo"]].apply(lambda x : x["combined_future_winner_elo"] if(x["Winner"] == player) else x["combined_future_loser_elo"], axis=1)
# a["538"] = a[["Winner","Loser","k_winner_240_5_0.4","k_loser_240_5_0.4"]].apply(lambda x : x["k_winner_240_5_0.4"] if(x["Winner"] == player) else x["k_loser_240_5_0.4"], axis=1)
# b["k"] = b[["Winner","Loser","k_winner_30","k_loser_30"]].apply(lambda x : x["k_winner_30"] if(x["Winner"] == player) else x["k_loser_30"], axis=1)


# a["elo"].plot(label="Surface Combined")
# a["538"].plot(label="Five Thirty Eight")
# b["k"].plot(label="K Factor")

# plt.title("30 Game Subset Elo Comparison")
# plt.xlabel("Game Number")
# plt.ylabel("Elo Score")

# plt.legend()
# plt.show()

#PROB PLOTS
a = pd.read_csv("output_models/temp_k_factor.csv")
player_plot_a = a.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
a["federer"] = player_plot_a[["Winner","Loser","k_winner_30","k_loser_30"]].apply(lambda x : x["k_winner_30"] if(x["Winner"] == player) else x["k_loser_30"], axis=1)
b = pd.read_csv("output_models/temp_538.csv")
player_plot_b = b.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
b["federer"] = player_plot_b[["Winner","Loser","k_winner_240_5_0.4","k_loser_240_5_0.4"]].apply(lambda x : x["k_winner_240_5_0.4"] if(x["Winner"] == player) else x["k_loser_240_5_0.4"], axis=1)

player_plot_c = df.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
player_plot_c["federer"] = player_plot_c[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo"]].apply(lambda x : x["combined_future_winner_elo"] if(x["Winner"] == player) else x["combined_future_loser_elo"], axis=1)

# a["federer"].plot(label="K Factor")
b["federer"].plot(label="Five Thirty Eight")
# player_plot_c["federer"].plot(label="Surface Combined")

plt.title("Federer Five Thirty Eight")
plt.xlabel("Game Number")
plt.ylabel("Elo Score")

plt.legend()
plt.show()

print(a["elo"])

#Federer Career Plot



# model = FiveThirtyEight()
# model = LogisticModel()
# # # surfaces = combined_dfs["Surface"].unique()
# # # model = SurfaceElo(surfaces)
# c = combined_dfs.reset_index()
# a = model.from_df(c)
# d = []

# d.append("prob")
# d.append("loser_prob")
# d.append("log_loss")
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
# b = a[e]

# b.to_csv("output_models/out_logistic.csv")

# # print(e)
# b = a[e]
# b.to_csv("output_models/out_538_prev.csv")

# b.to_csv("output_models/out_538.csv")
# d = []
# for surface in surfaces:
#     d.append('surface_' + surface+ "_winner_prev")
#     d.append('surface_' + surface+ "_loser_prev")
#     d.append('surface_' + surface+ "_winner")
#     d.append('surface_' + surface+ "_losprob_winnerer")
# e = ["Winner","Loser"] + d + ["", "prob_loser"]
# b = a[e]
# b.to_csv("output_models/out_surface_elo_prev.csv")
# a = pd.read_csv("output_models/out_logistic.csv",usecols=["Winner","Loser","prob","log_loss"])
# sum_gr_0_5_logistics = len(a[(a["prob"]>0.5)])
# print(f"accuracy log prob = {sum_gr_0_5_logistics/len(a)}")
# print(f"logloss log prob = {np.mean(a['log_loss'])}")

# a = pd.read_csv("output_models/out_538_prev.csv",usecols=["Winner","Loser","y_250_5_0.4","k_prev_winner_250_5_0.4","k_prev_loser_250_5_0.4","prob_250_5_0.4","loser_prob_250_5_0.4","log_loss_250_5_0.4"])
# # a.to_csv("output_models/out_538_optimal.csv")

# c = pd.read_csv("output_models/out_surface_elo_prev.csv",usecols=["prob_winner","prob_loser","surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"])

# joined = a.join(c)

# sigma = 0.85

# surface_fields = ["surface_Hard_winner_prev","surface_Hard_loser_prev","surface_Clay_winner_prev","surface_Clay_loser_prev","surface_Carpet_winner_prev","surface_Carpet_loser_prev","surface_Grass_winner_prev","surface_Grass_loser_prev"]


# joined["combined_prob"] = sigma*joined["prob_250_5_0.4"] + (1-sigma)*joined["prob_winner"]
# for index, row in joined.iterrows():   
#     for surface in surface_fields:
#         if not pd.isna(row[surface]) and "winner" in surface:
#             joined.loc[index,"combined_winner_elo"] = sigma*row["k_prev_winner_250_5_0.4"] + (1-sigma)*row[surface]
#         if not pd.isna(row[surface]) and "loser" in surface:    
#             joined.loc[index,"combined_loser_elo"] = sigma*row["k_prev_loser_250_5_0.4"] + (1-sigma)*row[surface]
        

# joined["combined_elo_prob"] = joined.apply(lambda row: pi_i_j(row["combined_winner_elo"],row["combined_loser_elo"]),axis=1)
# joined["sum_elo"] = joined["combined_winner_elo"] - joined["combined_loser_elo"]
# # print(joined["combined_elo_prob"])
# # joined.to_csv("output_models/combined_surface.csv")
# # joined = pd.read_csv("output_models/combined_surface.csv")
# # print(joined["prob_winner"])
# print(joined)
# print(len(joined[joined.sum_elo == 0.0]))
# joined = joined.drop(joined[joined.sum_elo == 0.0].index)
# joined = joined.reset_index()
# sum_gr_0_5_raw = len(joined[(joined["prob_250_5_0.4"]>0.5)])
# sum_gr_0_5 = len(joined[(joined["combined_prob"]>0.5)])
# sum_gr_0_5_comb = len(joined[(joined["combined_elo_prob"]>0.5)])

# print(f"accuracy comb prob = {sum_gr_0_5/len(joined)}")
# print(f"accuracy raw = {sum_gr_0_5_raw/len(joined)}")
# print(f"accuracy comb elo= {sum_gr_0_5_comb/len(joined)}")

# for index, row in joined.iterrows():    
#    if(row["combined_prob"] >0.5 ):
#        joined.loc[index,"combined_log_loss"] = test_ll(1.0,row["combined_prob"])
#        # df.loc[index,"loser_prob"] = 1-prob
#    else:
#        joined.loc[index,"combined_log_loss"] = test_ll(0.0,1.0-row["combined_prob"])

# for index, row in joined.iterrows():    
#    if(row["combined_winner_elo"] > row["combined_loser_elo"]):
#        joined.loc[index,"combined_log_loss_elo"] = test_ll(1.0,row["combined_elo_prob"])
#        joined.loc[index,"actual"] = 1.0
#        # df.loc[index,"loser_prob"] = 1-prob
#    else:
#        joined.loc[index,"combined_log_loss_elo"] = test_ll(0.0,1.0-row["combined_elo_prob"])
#        joined.loc[index,"actual"] = 0.0

# calbration_calc(joined,"combined_winner_elo","combined_loser_elo","combined_elo_prob")
# print(f"logloss comb prob = {np.mean(joined['combined_log_loss'])}")
# print(f"logloss = {np.mean(joined['log_loss_250_5_0.4'])}")
# print(f"logloss comb elo= {np.mean(joined['combined_log_loss_elo'])}")

# player = 'Federer R.'

# player_plot = a.query(f"Winner == '{player}' or Loser == '{player}'")
# player_plot = player_plot.reset_index()
# player_plot["elo"] = player_plot[["Winner","Loser","k_winner_260_5_0.4","k_loser_260_5_0.4"]].apply(lambda x : x["k_winner_260_5_0.4"] if(x["Winner"] == player) else x["k_loser_260_5_0.4"], axis=1)

# # player_plot["elo"] = player_plot[["Winner","Loser","k_winner_260_5_0.4","k_loser_260_5_0.4"]].apply(lambda x : x["k_winner_260_5_0.4"] if(x["Winner"] == player) else x["k_loser_260_5_0.4"], axis=1)
# print(player_plot[["Winner","Loser","elo"]])
# # player_plot["elo"].plot()
# # plt.show()
# # print(c)

# # # 0.0,0.4640522024570126,0.5359477975429874
# # # print(log_loss([0],[0.4640522024570126], labels=[0,]))
# a = pd.read_csv("output_models/out_k.csv")
# ranges = []
# accuracy = []
# ll = []
# for i in range(5,100+1,5):    
#     ranges.append(i)
#     sum_gr_0_5 = len(a[(a['prob_'+str(i)]>0.5)])
#     accuracy.append(sum_gr_0_5/len(a))
#     ll.append(np.mean(a['log_loss_'+str(i)]))
#     print(f"k = {i} accuracy = {sum_gr_0_5/len(a)}")
#     # logloss = log_loss(c['y'],c['prob'], labels=[1.0,0.0])
#     print(f"k = {i} logloss = {np.mean(a['log_loss_'+str(i)])}")

# # xs = [x for x in range(len(federer_k_list))]

# plt.plot(ranges, accuracy, label = "Accuracy")
# plt.plot(ranges, ll, label = "Log Loss")
# plt.legend()
# plt.show()
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

            


