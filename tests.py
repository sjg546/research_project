import os
import pandas as pd
class Tests():
    def handed_test(self,directory):
        player_hands = {}
        winner ={"right":0,"left":0}
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                # print(f)
                df = pd.read_csv(f)
                for ind in df.index:
                    contest_date=df["tourney_date"][ind]
                    winner_name = df["winner_name"][ind]
                    winner_hand = df["winner_hand"][ind]
                    loser_name = df["loser_name"][ind]
                    loser_hand = df["loser_hand"][ind]
                    if not pd.isna(winner_hand):
                        if not winner_name in player_hands:
                            player_hands[winner_name] = winner_hand
                    if not pd.isna(loser_hand):
                        if not loser_name in player_hands:
                            player_hands[loser_name] = loser_hand

                    # ratings = parser.get_current_rankings(str(contest_date))
                    if not pd.isna(winner_hand) or not pd.isna(loser_hand):
                            if winner_hand == "R" and loser_hand == "L":
                                winner["right"] += 1
                            elif winner_hand == "L" and loser_hand == "R":
                                winner["left"] += 1
                                 
        right = 0
        left = 0
        for key in player_hands:
            if player_hands[key] == "R":
                right += 1
            elif player_hands[key] == "L":
                left += 1
        print("-------ELO RESULTS-------")
        print(winner)
        print("Right:" +str(right)+", Left:" +str(left))
        pct = winner["right"]/ (winner["right"] + winner["left"])
        print(pct)
        print("-------------------------")

    def height_test(self,directory):
        player_heights = {}
        winner ={"tall":0,"short":0}
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                # print(f)
                df = pd.read_csv(f)
                for ind in df.index:
                    contest_date=df["tourney_date"][ind]
                    winner_name = df["winner_name"][ind]
                    winner_height = df["winner_ht"][ind]
                    loser_name = df["loser_name"][ind]
                    loser_height = df["loser_ht"][ind]
                    if not pd.isna(winner_height):
                        if not winner_name in player_heights:
                            player_heights[winner_name] = winner_height
                    if not pd.isna(loser_height):
                        if not loser_name in player_heights:
                            player_heights[loser_name] = loser_height

                    # ratings = parser.get_current_rankings(str(contest_date))
                    if not pd.isna(winner_height) or not pd.isna(loser_height) and ((winner_height > 183.8 and loser_height <183.8) or (winner_height < 183.8 and loser_height > 183.8)):
                            if winner_height > loser_height:
                                winner["tall"] += 1
                            else:
                                winner["short"] += 1
        total = 0
        for key in player_heights:
              total += player_heights[key]
            
        average = total/len(player_heights) 
        print(average)                 
        print("-------ELO RESULTS-------")
        print(winner)
        pct = winner["tall"]/ (winner["tall"] + winner["short"])
        print(pct)
        print("-------------------------")

