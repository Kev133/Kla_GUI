import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


kla=0.5
rozpO2=1.396 # u Labika jako alfa
rozpN2=0.6817
temp_vsadky = 20
mO2 = rozpO2*(273.15+temp_vsadky)*8.314472/101325
mN2 = rozpN2*(273.15+temp_vsadky)*8.314472/101325
difO2=2.11e-9
difN2=1.74e-9
V_kapalina=158.4
y = 0.21 #v práci labika vic do detailu, zatim nechám takto

A =8.07131
B =-1730.63
C =233.426
pH2O = 10**(A+B/(temp_vsadky+C))*101325/760 #Antoinova rovnice mi totiz vraci tlak v mmHg
prutok_plynu = 36
frekvence_mich = 250
zadrz_plynu = 0.0088
prikon_michadla = 52.889
deltaVg = V_kapalina*zadrz_plynu/(1-zadrz_plynu)
pO2_G = 21000
pN2_G = 78000
pG = pO2_G + pN2_G + pH2O
pG_ust1 = 101000  #prvni hodnota na vstupu, zacatek experimentu v ***.dtm
pG_ust2 = 151000  #random hodnota z ustalene casti kdy je tlak maximalni
xG = (pG-pG_ust2)/(pG_ust1-pG_ust2)
print(xG)
print(pH2O)
def dSdt(t,S):
    xO2L,xN2L,xO2G = S
    return([kla*(xO2G-xO2L),
            kla*2*(xO2G*5-xN2L),
            6*0.2-2*kla*(xO2G-xO2L)-10*xO2G])
# počáteční podmínky
xO2L_0 = 1
xO2G_0 = 1
xN2L_0 = 1
S_0=(xO2L_0,xN2L_0,xO2G_0)

t = np.linspace(0,1,50)
sol = odeint(dSdt,y0=S_0,t=t,tfirst=True)

O2L=sol[:,0]
N2L=sol[:,1]
O2G=sol [:,2]


plt.plot(t,O2L)
plt.plot(t,N2L)
plt.plot(t,O2G)
plt.show()