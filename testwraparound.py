import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

pi = np.pi
exp = np.exp
Km1 = 1.052082 / (2 * pi ** 2)
N = 1000
cas_body =3101
t = np.linspace(0, 155, num=cas_body)
n = np.linspace(0, 1000, num=1001)

It_Opt = np.zeros_like(t)  # create an array to store the results
Ht_Opt = np.zeros_like(t)
for i, ti in enumerate(t):
    It_Opt[i] = np.sum(-8 * exp(-pi ** 2 * Km1 * ti * ((2 * n + 1) ** 2) / 4) * (
                (1 / ((2 * n + 1) ** 2 * pi ** 2)) * ((-pi ** 2 * Km1 * (2 * n + 1) ** 2) / 4)))
for i,ti in enumerate(t):
    Ht_Opt [i] = np.sum (1-8*exp(-pi**2*Km1*ti*(2*n+1)**2/4)*(1/(2*n+1)**2*pi**2))
Ht_Opt_N = (Ht_Opt - Ht_Opt.max()) / (Ht_Opt.min() - Ht_Opt.max())
char_sondy = []
for i in range(0, len(It_Opt)):
    char_sondy.append(
        (It_Opt[i] - min(It_Opt)) / (max(It_Opt) - min(It_Opt)))

h_orig = Ht_Opt_N
char_sondy = Ht_Opt_N
X= np.linspace(0, 155, num=cas_body)
b = -0.00024*X**2+0.3501*X+925.72
min = 930
max = 976
x1 = (b - max) /( min - max)
num_ones=int(cas_body/3)
num_zeros=int(cas_body/10)
x=np.concatenate((np.ones(num_ones),x1))
x=np.concatenate([x,np.zeros(num_zeros)])
char_sondy_otocena=np.flip(char_sondy)
char_sondy_uprav=np.concatenate((char_sondy_otocena[cas_body-num_ones:cas_body],char_sondy))
char_sondy_uprav=np.concatenate([char_sondy_uprav,np.zeros(num_zeros)])

# # Define length of signal and original impulse response
# L = len(x)
# N = len(h_orig)
#
# # Create wraparound version of impulse response
# h_wrap = np.zeros(L + N - 1)
# h_wrap[:N] = h_orig
# h_wrap[N-1:] = h_orig[-1:-(L+N):-1]
#
# # Convolve wraparound impulse response with signal
y = np.convolve(x, Ht_Opt_N)

# Plot results
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

axs[0].plot(h_orig, 'o-', markersize=4)
axs[0].set_title('Original impulse response')

axs[1].plot(char_sondy_uprav, 'o-', markersize=4)
axs[1].set_title('Wraparound impulse response')

axs[2].plot(x, 'o-', markersize=4)
axs[2].set_title('Input signal')

axs[3].plot(y, 'o-', markersize=4)
axs[3].set_title('Output signal')

plt.tight_layout()
plt.show()