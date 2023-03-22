import matplotlib.pyplot as plt
import numpy as np
pi = np.pi
exp = np.exp
def custom_float(value):
    try:
        return float(value)
    except ValueError:
        return float(value.replace("d", "e"))


# loading data from konstant.dta
with open("konstant.DTA") as f:
    konstant = f.readlines()

header = konstant[0]
# je tam mezera před nazvem tak pouzivam strip
probe_name = konstant[1].strip()
Km1, Km2, Zg1 = map(float, konstant[2].split())
Km1 = Km1/(pi**2)
print(Km1)
mO2_raw, mN2_raw, difO2, difN2 = map(custom_float, konstant[3].split())
volume =  float(konstant[4])
print(volume)
volume = volume/1000
values = []

N = 1000 #takhle to ma labik, zatim necham stejne


t=0
one = 0
It_Opt = 0
#Ht = 1+2*Zg1*exp(-pi**2*Km1*t*n**2)*(-1)**n+2*Zg2*exp(-pi**2*Km2*t*n**2)*(-1)**n

      #It = 2*Zg1*exp(-pi**2*Km1*t*n**2)*(-pi**2*Km1*n**2)*(-1)**n  +  2*Zg2*exp(-pi**2*Km2*t*n**2)*(-pi**2*Km2*n**2)*(-1)**n
t= np.linspace(0,155,num=3101)
n= np.linspace(0,1000,num=1001)

It_Opt = np.zeros_like(t)  # create an array to store the results

for i, ti in enumerate(t):
    It_Opt[i] = np.sum(-8 * exp(-pi**2 * Km1 * ti * ((2*n+1)**2)/4) * ((1/((2*n+1)**2*pi**2))*((-pi**2*Km1*(2*n+1)**2)/4)))

print(It_Opt)
#plt.plot(t,It_Opt)

# It_OptN = []
# for i in range(0, len(It_Opt)):
#     It_OptN.append(
#         (It_Opt[i] - min(It_Opt)) / (max(It_Opt) - min(It_Opt)))


plt.plot(t,It_Opt)
plt.show()