from player import Player
import math
from scipy import optimize

class ELOMMR:

    def __init__(self):
        self.mean_init = 1500
        self.std_dev_int = 350
        self.player_list:dict = {}
        self.round_list:list = []
        self.beta = 0.2
        self.p = 1
        self.y = 80

    def calculate_round(self, players:list[int]):
        self.round_list.clear()
        ss = []
        ps = []
        ts = []
        ds = []
        for player in players:
            if not player in self.player_list:
                    self.player_list[player] = Player(player, self.mean_init, self.std_dev_int)
                    
            
            self.diffuse(self.player_list[player])
            current_player:Player = self.player_list[player]

            current_player.mean_pi = current_player.mean
            current_player.delta = math.sqrt(math.pow(current_player.std_dev,2)+math.pow(self.beta,2))

            self.round_list.append(self.player_list[player])
        k = 0
        for player in players:
            self.update(self.player_list[player],k)
        

    def diffuse(self, player:Player):
       
        k =  math.pow(1 +(math.pow(self.y,2)/math.pow(player.std_dev,2)),-1)

        wg = math.pow(k,self.p) * player.w[0]
        sum_w = 0.0 
        for w in player.w:
            sum_w += w

        wl = (1-math.pow(k,self.p))*sum_w

        player.p[0] = float((wg*player.p[0] + wl*player.mean)/(wg+wl))
        player.w[0] = float((k*(wg+wl)))
        for i in range(len(player.w)):
            player.w[i] = math.pow(k,(1+self.p)) * player.w[i]


        player.std_dev = player.std_dev/math.sqrt(k)
        # root_scalar(objective_function, bracket=[0.0, 0.1])


    def f1(self,x,player:Player, pos:int):
        suma = 0
        sumb = 0

        for j in range(len(self.round_list)):
            if j <= pos:
                suma += (1/player.delta) * (math.tanh((x-player.mean_pi)/(2*player.delta))-1)
            if j >= pos:
                sumb += (1/player.delta) * (math.tanh((x-player.mean_pi)/(2*player.delta))+1)     

        return suma + sumb
         
    def f2(self,x,player:Player):
        a = player.w[0]*(x-player.p[0])
        sum_w = 0.0
        for i in range(len(player.w)):    
            sum_w += (((player.w[i]*math.pow(self.beta,2))/self.beta))* math.tanh((x-player.p[i])/(2*self.beta))
        return a + sum_w
    
    def update(self,player:Player,k:int):
        p = optimize.root_scalar(self.f1, x0=0.2, bracket=[-10000.0, 9000.0], args = (player,k),  method='brentq')
        player.p.append(p.root) #output of weird stuff
        player.w.append(1/math.pow(self.beta,2)) #done
        player.mean = optimize.root_scalar(self.f2, x0=0.2, bracket=[-10000.0, 9000.0], args = (player),  method='brentq').root
    