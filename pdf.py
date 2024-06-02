import math
class PDF:
     @staticmethod
     def pdf(a:float):
        inv_sqrt_2pi = 0.3989422804014327
        return inv_sqrt_2pi * math.exp(-0.5 * a * a)
