import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt



#TODO neslo by pridat i kLa ktere se pak teda najde po optimalizaci? Bylo by to zajimavy tak nakodovat
# a alspon vic motivace pro me, spis ale spare time vec

class Optimalizace():
    def __init__(self,impulse,namerene):
        self.impulse= impulse
        self.namerene= namerene

        self.vysledek = self.opt()


    def to_opt(self,x):
        return sum((np.convolve(x, self.impulse) - self.namerene) ** 2)



    def opt(self):
        x0=[0.5]*150
        return scipy.optimize.minimize(self.to_opt, x0)


# vykresleni do grafu
    def graph(self):
        fig, axes = plt.subplots(1, 1)
        axes.plot(self.namerene[0:int(len(self.namerene)/2)], marker=".", label="Naměřené hodnoty", color='tab:red')
        axes.plot(self.vysledek.x, marker=".", label="Odhah po optimalizaci", color='tab:green')
        axes.plot(self.impulse, marker=".", label="Impulse", color='tab:orange')
        axes.legend(['Naměřené hodnoty', 'Odhad po optimalizaci', "impulzní charakteristika"])
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        fig.suptitle('Optimalizační model')

        plt.show()

