import os
import json
class ResultsParser():
    def __init__(self):
        self.ratings_at_date = {}
        self.keys = []
        self.values = []

    def read_ratings(self):
        for filename in os.listdir('results\output'):
            with open('results\\output\\' + filename) as f:
                print(filename)
                test: str = str(f.read())
                test2 = test[:-3] +"}"
                tournament_date = filename[:-4]
                self.ratings_at_date[tournament_date] = json.loads(test2)
        
        self.keys = list(self.ratings_at_date.keys())
        self.values = list(self.ratings_at_date.values())


    def get_current_rankings(self, date:str):
        if date in self.keys:
            res = self.keys.index(date) -1
            if res < 0:
                return {}
            value = self.values[res]
            return value
        return {}

            # callthecommandhere(blablahbla, filename, foo)
