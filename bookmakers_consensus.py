import pandas as pd
import math
import numpy as np
class BookmakersConsensus():
    def __init__(self) -> None:
        a=1
        self._betting_companies = ['B365','B&W','CB','EX','LB','GB','IW','PS','SB','SJ','UB']
        self._betting_companies_w_l = ['B365W','B365L','B&WW','B&WL','CBW','CBL','EXW','EXL','LBW','LBL','GBW','GBL','IWW',
        'IWL','PSW','PSL','SBW','SBL','SJW','SJL','UBW','UBL']
        # self._winning
        self._column_drop_list = ['Round','Best of','W1','L1','W2','L2','W3','L3','W4','L4','W5','L5','Wsets','Lsets']
    def clean_data(self, csv):
        df = pd.read_csv(csv)
        narrowed_companies_list = []
        narrowed_drop_list = []
        for company in self._betting_companies_w_l:
            if company in df:
                narrowed_companies_list.append(company)
        for field in self._column_drop_list:
            if field in df:
                narrowed_drop_list.append(field)
        # Drop a bunch of fields i dont care about
        df = df.drop(narrowed_drop_list, axis=1)
        # Drop rows which have no odds from any company
        df_cleaned = df.dropna(subset=narrowed_companies_list, how='all')
        # df_cleaned["Comment"] = pd.np.where(if df_cleaned.Comment.str.contains("Completed"), "Completed","NA")    
        # df_cleaned = df_cleaned[df_cleaned.Comment == "Completed"]
        # print(df_cleaned)
        return df_cleaned
    def test_ll(self,y,prob):
        if y == prob:
            return 0.0
        if y == 0.0 and prob == 1.0:
            return 0.0
            
        return -1* ((y * math.log(prob)) + ((1-y)*math.log(1-prob)))

    def calculate_odds(self, df:pd.DataFrame):
        logitp1 = []
        logitp2 = []
        companies_used = []
        temp = []
        for company in self._betting_companies:
            if company +"W" in df:
                company_str = str(company)
                col_name_p1 = company_str+"P1"
                col_name_p2 = company_str+"P2"
                winner = company_str +"W"
                loser = company_str+"L"
                df[col_name_p1] = df[loser]/(df[winner]+df[loser])
                df[col_name_p2] = df[winner]/(df[loser]+df[winner])
                df["logit" + col_name_p1] = np.log(df[col_name_p1]/(1-df[col_name_p1]))
                df["logit" + col_name_p2] = np.log(df[col_name_p2]/(1-df[col_name_p2]))
                # companies_used.append(winner)
                # companies_used(loser)
                logitp1.append("logit"+col_name_p1)
                logitp2.append("logit"+col_name_p2)
               
        df['logitP1'] = df[logitp1].mean(axis=1)
        df['logitP2'] = df[logitp2].mean(axis=1)
        df['p1'] = (np.e**df['logitP1'])/(1+(np.e**df['logitP1']))
        df['p2'] = (np.e**df['logitP2'])/(1+(np.e**df['logitP2']))

        df = df.dropna(subset=["p1","p2"])
        df = df.reset_index()
        
    def run_metrics(self,df:pd.DataFrame):
        df = df.dropna(subset=["p1","p2"])
        df = df.reset_index()
        df["actual"] = 1.0
        for index, row in df.iterrows():
           if(row["p1"] > 0.5 ):
               df.loc[index,"combined_log_loss"] = self.test_ll(1.0,row["p1"])

           else:
               df.loc[index,"combined_log_loss"] = self.test_ll(0.0,row["p2"])

           if(row["p1"] < 0.5 ):
              df.loc[index,"predicted"] = 0.0
           else:
              df.loc[index,"predicted"] = 1.0

        total_prob = np.sum(df[f"p1"])
        print(f"BM Calibration = {total_prob/len(df[df.p1 > 0.5])}")
        print(f"BM Accuracy = {len(df[df.p1 > 0.5])/len(df['p1'])}")
        print(f"BM Logloss = {np.mean(df['combined_log_loss'])}")
            

        
