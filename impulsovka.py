import matplotlib.pyplot as plt
import numpy as np

# 1LTN voda
pi = np.pi
exp = np.exp
Km1 = 1.052082/(pi**2)
Km2 = 0
Zg1 = 1
Zg2= 1-Zg1
rozpO2=1.396
rozpN2=0.6817
#nutno prepocitat rozp na m podle mO = rozp_O2*(273.15+teplota_vsadky)*8.314472/101325 stejne tak pro N

difO2=2.11e-9
difN2=1.74e-9
V_kapalina=158.4
N = 1000 #takhle to ma labik, zatim necham stejne
n = np.linspace(1, N,num=N)
t = np.linspace(0,155,num=N)


prutok_plynu = 36
frekvence_mich = 250
zadrz_plynu = 0.0088
prikon_michadla = 52.889
deltaVg = V_kapalina*zadrz_plynu/(1-zadrz_plynu)
Ht = 1+2*Zg1*exp(-pi**2*Km1*t*n**2)*(-1)**n+2*Zg2*exp(-pi**2*Km2*t*n**2)*(-1)**n
It = 2*Zg1*exp(-pi**2*Km1*t*n**2)*(-pi**2*Km1*n**2)*(-1)**n  +  2*Zg2*exp(-pi**2*Km2*t*n**2)*(-pi**2*Km2*n**2)*(-1)**n

plt.plot(t,It)
with open("C:/Users/Kevin/Desktop/example_data/namerene_hodnoty.txt", "r") as f:
    hodnoty1 = f.read().splitlines()
namerene = list(map(float, hodnoty1))
print(namerene)