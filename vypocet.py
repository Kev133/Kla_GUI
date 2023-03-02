import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
def to_opt(kla,impulse,sonda):

    rozpO2 = 1.396  # u Labika jako alfa
    rozpN2 = 0.6817
    temp_vsadky = 20
    mO2 = rozpO2 * (273.15 + temp_vsadky) * 8.314472 / 101325
    mN2 = rozpN2 * (273.15 + temp_vsadky) * 8.314472 / 101325
    difO2 = 2.11e-9
    difN2 = 1.74e-9
    V_kapalina = 158.4 / 1000
    y = 0.21

    A = 8.07131
    B = -1730.63
    C = 233.426
    pH2O = 10 ** (A + B / (temp_vsadky + C)) * 101325 / 760  # Antoinova rovnice mi totiz vraci tlak v mmHg

    prutok_plynu_in = 6e-4
    zadrz_plynu = 0.0088
    deltaVg = V_kapalina * zadrz_plynu / (1 - zadrz_plynu)
    with open("C:/Users/Kevin/Desktop/example_data/namerene_hodnoty.dtm", "r") as f:
        hodnoty1 = f.read().splitlines()
    namerene = list(map(float, hodnoty1))
    pGraw = np.array(namerene[16:416])

    t = np.array(namerene[1622:1622 + 400])

    pG_ust1 = 748 * 133.3  # prvni hodnota na vstupu, zacatek experimentu v ***.dtm
    pG_ust2 = max(pGraw) + pG_ust1
    p1 = pGraw[0]
    p2 = max(pGraw)

    def savitzky_golay_filter(data, window_size, order):
        return savgol_filter(data, window_size, order)

    xG = (pGraw - p2) / (p1 - p2)
    xG = savitzky_golay_filter(xG, window_size=11, order=2)

    cs = CubicSpline(t, xG)
    t = np.linspace(0, max(t), num=3101)

    print(kla)


    def dSdt(t, S):
        xO2L, xN2L, xO2G = S

        dxO2L = kla * (xO2G - xO2L)
        dxN2L = kla * (difN2 / difO2) ** (0.5) * ((cs(t) - xO2G * y) / (1 - y) - xN2L)

        prutok_plynu_out = (prutok_plynu_in * (pG_ust2 - pH2O) / (pG_ust1 - pG_ust2) - V_kapalina * (
                mO2 * dxO2L * y + mN2 * dxN2L * (1 - y)) - deltaVg * cs.derivative()(t)) / (
                                       cs(t) + (pG_ust2 - pH2O) / (pG_ust1 - pG_ust2))

        dxO2G = (prutok_plynu_in * (pG_ust2 - pH2O) / (
                pG_ust1 - pG_ust2) - V_kapalina * mO2 * dxO2L - prutok_plynu_out * (xO2G + (

                pG_ust2 - pH2O) / (pG_ust1 - pG_ust2))) / deltaVg

        return np.array([dxO2L,
                 dxN2L,
                 dxO2G]).flatten()

    # počáteční podmínky
    xO2L_0 = 1
    xO2G_0 = 1
    xN2L_0 = 1
    S_0 = (xO2L_0, xN2L_0, xO2G_0)


    sol = odeint(dSdt, y0=S_0, t=t, tfirst=True)

    O2L = sol[:, 0]
    N2L = sol[:, 1]
    O2G = sol[:, 2]

    prubeh = (O2L - O2L.min()) / (O2L.max() - O2L.min())
    #plt.plot(t,prubeh)
    # plt.plot(t, impulse)
    # plt.plot(t, sonda)

    plt.plot(t,(np.convolve(prubeh, impulse))[0:3101])
    # plt.show()
    return sum((np.convolve(prubeh, impulse)[0:3101] -sonda) ** 2)



def opt(choice,impulse,namerene):
    x0=0.001

    if choice == 1:
        return scipy.optimize.minimize(to_opt, x0,args=(impulse,namerene), method ="Nelder-Mead").x
    elif choice == 2:
        return scipy.optimize.minimize(to_opt, x0,method ="BFGS").x
    elif choice == 3:
        return scipy.optimize.minimize(to_opt, x0,method ="Powell").x
    else:
        print(choice)

if __name__=="__main__":
    pass