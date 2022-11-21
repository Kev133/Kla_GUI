import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
import time





#TODO neslo by pridat i kLa ktere se pak teda najde po optimalizaci? Bylo by to zajimavy tak nakodovat
# a alspon vic motivace pro me, spis ale spare time vec

class Optimalizace():
    def __init__(self,impulse,namerene):
        self.impulse= impulse
        self.namerene= namerene
        self.namereneN=[]
        #self.vysledek = self.opt()
        self.funkce = []

    def to_opt(self,kla):
        self.funkce = []
        tau= np.linspace(0, 100, num=1000)
        self.hod = np.exp(-kla*tau)

        values = self.hod
        for i in range(0, len(values)):
            self.funkce.append(
                (values[i] - max(values)) / (min(values) - max(values)))
        return sum((np.convolve(self.funkce, self.impulse) - self.namerene) ** 2)



    def opt(self,choice):
        x0 = 0.1
        if choice == 1:
            return scipy.optimize.minimize(self.to_opt, x0,method ="Nelder-Mead").x
        elif choice == 2:
            return scipy.optimize.minimize(self.to_opt, x0,method ="BFGS").x
        elif choice == 3:
            return scipy.optimize.minimize(self.to_opt, x0,method ="Powell").x
        else:
            print(choice)
# vykresleni do grafu, nepouzivane
    def graph(self):
        tau = np.linspace(0, 100, num=500)
        self.hod = np.exp(-0.1879 * tau)
        values = self.hod

        for i in range(0, len(values)):
            self.funkce.append(
                (values[i] - max(values)) / (min(values) - max(values)))



        for i in range(0, len(self.namerene)):
            self.namereneN.append(
                (self.namerene[i] - min(self.namerene)) / (max(self.namerene) - min(self.namerene)))

        fig, axes = plt.subplots(1, 1)
        axes.plot(self.namereneN[0:int(len(self.namereneN)/2)], marker=".", label="Naměřené hodnoty", color='tab:red')
        axes.plot(self.funkce, marker=".", label="Odhah po optimalizaci", color='tab:green')
        axes.plot(self.impulse, marker=".", label="Impulse", color='tab:orange')
        axes.legend(['Naměřené hodnoty', 'Odhad po optimalizaci', "impulzní charakteristika"])
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        fig.suptitle('Optimalizační model')
        plt.show()






if __name__=="__main__":
    pass