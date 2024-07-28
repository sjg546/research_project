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

'''
0 = No Plot
1 = 30 Game Subset
2 = Federer Career Plot
NOTE, if set to anything other than 0, MODEL_TO_USE will be set to 4
'''
GENERATE_PLOT=0

K_VALUE = 30

FIVE_THIRTY_EIGHT_DELTA = 240
FIVE_THIRTY_EIGHT_SIGMA = 0.4
FIVE_THIRTY_EIGHT_NU = 5

SURFACE_SIGMA = 0.85

directory = "tennis_atp/mens_atp"
    
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


if not GENERATE_PLOT == 0:
     MODEL_TO_USE = 4

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
if GENERATE_PLOT == 1:
    player_plot = df.query(f"Winner == '{player}' or Loser == '{player}'")
    player_plot = player_plot.reset_index()
    b = pd.read_csv("output_models/temp_k_factor.csv")
    player_plot2 = b.query(f"Winner == '{player}' or Loser == '{player}'")
    player_plot2 = player_plot2.reset_index()

    a = player_plot.iloc[300:330]
    b = player_plot2.iloc[300:330]
    c = a.merge(b)
    print(c[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo","k_winner_240_5_0.4","k_loser_240_5_0.4","k_winner_30","k_loser_30"]])
    a["elo"] = a[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo"]].apply(lambda x : x["combined_future_winner_elo"] if(x["Winner"] == player) else x["combined_future_loser_elo"], axis=1)
    a["538"] = a[["Winner","Loser","k_winner_240_5_0.4","k_loser_240_5_0.4"]].apply(lambda x : x["k_winner_240_5_0.4"] if(x["Winner"] == player) else x["k_loser_240_5_0.4"], axis=1)
    b["k"] = b[["Winner","Loser","k_winner_30","k_loser_30"]].apply(lambda x : x["k_winner_30"] if(x["Winner"] == player) else x["k_loser_30"], axis=1)


    a["elo"].plot(label="Surface Combined")
    a["538"].plot(label="Five Thirty Eight")
    b["k"].plot(label="K Factor")

    plt.title("30 Game Subset Elo Comparison")
    plt.xlabel("Game Number")
    plt.ylabel("Elo Score")

    plt.legend()
    plt.show()

#PROB PLOTS
if GENERATE_PLOT == 2:
    a = pd.read_csv("output_models/temp_k_factor.csv")
    player_plot_a = a.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
    a["federer"] = player_plot_a[["Winner","Loser","k_winner_30","k_loser_30"]].apply(lambda x : x["k_winner_30"] if(x["Winner"] == player) else x["k_loser_30"], axis=1)
    b = pd.read_csv("output_models/temp_538.csv")
    player_plot_b = b.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
    b["federer"] = player_plot_b[["Winner","Loser","k_winner_240_5_0.4","k_loser_240_5_0.4"]].apply(lambda x : x["k_winner_240_5_0.4"] if(x["Winner"] == player) else x["k_loser_240_5_0.4"], axis=1)

    player_plot_c = df.query(f"Winner == '{player}' or Loser == '{player}'").reset_index()
    player_plot_c["federer"] = player_plot_c[["Winner","Loser","combined_future_winner_elo","combined_future_loser_elo"]].apply(lambda x : x["combined_future_winner_elo"] if(x["Winner"] == player) else x["combined_future_loser_elo"], axis=1)

    a["federer"].plot(label="K Factor")
    b["federer"].plot(label="Five Thirty Eight")
    player_plot_c["federer"].plot(label="Surface Combined")

    plt.title("Federer Career Elo")
    plt.xlabel("Game Number")
    plt.ylabel("Elo Score")

    plt.legend()
    plt.show()