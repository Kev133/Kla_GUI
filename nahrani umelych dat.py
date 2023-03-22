

 # TODO skript na vytvoření umělých dat = impulse response a namerene hodnoty, ve zvolene složce (dole ve scriptu se dá přejmenovat adresa)
 # N znamena normalizovany

import numpy as np
measured_valuesN = []
real_valuesN = []
guessN = []
impulse_responseN=[]
# zavedeni promenne tau
tau = np.linspace(0, 15, num=400)
# nastrel x0 pro optimalizaci
x0 = len(tau) * [0.5]
# charakteristika sondy
#impulse_response = np.linspace(0, 3, num=kolik_bodu)
impulse_response= np.exp(-1.05*tau)*(-1.05)
for i in range (0,len(impulse_response)):
    impulse_responseN.append(
        (impulse_response[i]-max(impulse_response))/(min(impulse_response)-max(impulse_response)))

# normalizace opravdových hodnot
kla=0.701
real_values = np.exp(-kla*tau)
for i in range(0, len(real_values)):
    real_valuesN.append(
        (real_values[i] - max(real_values)) / (min(real_values) - max(real_values)))

# tvorba namerenych hodnot = konvoluce real_hodnot a impulsni charakteristiky

measured_values=np.convolve(real_valuesN, impulse_responseN)
print(len(real_valuesN))
print(len(impulse_response))
print(len(measured_values))


# normalizace measured values
for i in range (0,len(measured_values)):
     measured_valuesN.append(
        (measured_values[i]-min(measured_values))/(max(measured_values)-min(measured_values)))
#ulozeni dat do slozek pro GUI

# with open( "C:/Users/Kevin/Desktop/program/example_data/namerene_hodnoty.txt", 'w') as f:
#     for line in measured_values:
#         f.write(f"{line}\n")
with open( "C:/Users/Kevin/Desktop/example_data/konstant.txt", 'w') as f:
    for line in impulse_responseN:
        f.write(f"{line}\n")