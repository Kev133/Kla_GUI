import matplotlib.pyplot as plt
import numpy as np

# 1LTN voda
pi = np.pi
exp = np.exp
Km1 = 1.052082/(2*pi**2)
Km2 = 0
Zg1 = 1
Zg2= 1-Zg1


N = 1000 #takhle to ma labik, zatim necham stejne

t= np.linspace(0,155,num=3101)


one = 0
It_Opt = 0
#Ht = 1+2*Zg1*exp(-pi**2*Km1*t*n**2)*(-1)**n+2*Zg2*exp(-pi**2*Km2*t*n**2)*(-1)**n

      #It = 2*Zg1*exp(-pi**2*Km1*t*n**2)*(-pi**2*Km1*n**2)*(-1)**n  +  2*Zg2*exp(-pi**2*Km2*t*n**2)*(-pi**2*Km2*n**2)*(-1)**n
for n in range(0,1001):
    print (n)
    two = -8 * exp(-pi**2 * Km1 * t * (2*n+1)**2/4) * ((1/((2*n+1)**2*pi**2))*(-pi**2*Km1*(2*n+1)**2/4))
    clen = one + two
    It_Opt = It_Opt + clen
    one = two

#plt.plot(t,It_Opt)

It_OptN = []
for i in range(0, len(It_Opt)):
    It_OptN.append(
        (It_Opt[i] - min(It_Opt)) / (max(It_Opt) - min(It_Opt)))

with open("C:/Users/Kevin/Desktop/example_data/namerene_hodnoty.txt", "r") as f:
    hodnoty1 = f.read().splitlines()
namerene = list(map(float, hodnoty1))
plt.plot(t,It_OptN)
plt.show()