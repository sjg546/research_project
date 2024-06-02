import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
directory = "tennis_atp/mens_atp"
win_count = 0
lose_count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # print(f)
        df = pd.read_csv(f)
        for ind in df.index:
            if not pd.isnull(df['winner_rank'][ind]) and not pd.isnull(df['loser_rank'][ind]):
                if df['winner_rank'][ind] < df['loser_rank'][ind]:
                    win_count += 1
                else:
                    lose_count += 1

total = win_count + lose_count
percent_win = (win_count / total)*100
data = [["Games Played", total], ["Higher Rank", win_count], ["Lower Rank", lose_count], ["Percentage", percent_win] ]
col_names = ["Field", "Value"]
print(tabulate(data, headers=col_names))

# print(percent_win)
# print(win_count)
# print(lose_count)



# df = pd.read_csv('tennis_atp/atp_matches_2000.csv')

# # print(df[df["winner_name"] == "Tommy Haas"]["tourney_date"])
# # df["tourney_date"] = pd.to_datetime(df["tourney_date"],format='%Y%m%d')
# # winning_df = df[df["winner_name"] == "Tommy Haas"]
# # losing_df = df[df["loser_name"] == "Tommy Haas"]
# # losing_view=losing_df[["loser_rank","tourney_date"]]
# # winning_view=winning_df[["winner_rank","tourney_date"]]
# # frames = [losing_view,winning_view]
# # combined_view=pd.concat(frames)
# win_count = 0
# lose_count = 0
# for ind in df.index:
#     if df['winner_rank'][ind] > df['loser_rank'][ind]:
#         win_count += 1
#     else:
#         lose_count += 1

# total = win_count + lose_count
# percent_win = win_count / total
# print(percent_win)

# print(combined_view)
# print(losing_df[["loser_rank","tourney_date"]])
# print(winning_df[["winner_rank","tourney_date"]])

# ser = pd.Series([1, 2, 3, 3])
# plot = plt.plot(df["winner_rank"])
# plt.savefig('tommy_haas.png')