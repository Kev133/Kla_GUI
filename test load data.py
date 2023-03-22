"""his module loads data from the files "konstant.dta" and "xxxx.dtm"
it also contains the model and optimization function.
"""

# inc and dec are used for the words increase and decrease
import glob
import scipy
import numpy as np
from scipy.integrate import odeint, simps
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
pi= np.pi
exp = np.exp
def custom_float(value):
    try:
        return float(value)
    except ValueError:
        return float(value.replace("d", "e"))

# loading data from paramy.dta
with open("paramy.DTA") as f:
    paramy = f.readlines()
manometer_constant = custom_float(paramy[0])
spline_accuracy = custom_float(paramy[1])
upper_limit, lower_limit = map(custom_float,paramy[2].split())
kla_accuracy = custom_float(paramy[3])
# loading data from konstant.dta
with open("konstant.DTA") as f:
    konstant = f.readlines()

header = konstant[0]
# je tam mezera před nazvem tak pouzivam strip
probe_name = konstant[1].strip()
Km1, Km2, Zg1 = map(float, konstant[2].split())
mO2_raw, mN2_raw, difO2, difN2 = map(custom_float, konstant[3].split())
volume = float(konstant[4])/1000
values = []

for line in konstant[5:]:
    values.append(tuple(map(float, line.split())))
#TODO in the first [], put the number of he experiment you want to evaluate -1,
# this example is for the 1LT16A experiment
gas_in2, h, gas_hold_up, agitator_power = values[5][0:4]

gas_in2 = gas_in2/1000/60
# loading data from xxxx.dtm, have to add [0] because it returns a list
# TODO here in the [] brackets, put the number of the experiment you want to evaluate,-1
dtm_file = glob.glob("*.dtm")[5]
print(dtm_file)
with open(dtm_file, "r") as f:
    exp_profiles = f.read().splitlines()

exp_profiles = list(map(float, exp_profiles))
temp = exp_profiles[2]
pG_atm = exp_profiles[3] * 101325/760

gas_in = exp_profiles[4]/1000/60
agitator_frequency = exp_profiles[5]
num_of_channels = exp_profiles[6]
y = exp_profiles[7]
measurement_date = exp_profiles[8]
measurement_time = exp_profiles[9]
num_data_inc = int(exp_profiles[10])
num_data_dec = int(exp_profiles[11])
steady_pG_1 = exp_profiles[12]

current_line = 13

pG_data_inc = exp_profiles[current_line:num_data_inc + current_line]

current_line = current_line + num_data_inc

steady_pG_2 = exp_profiles[current_line+1]

steady_pG_2_down = exp_profiles[current_line + 1] # unused

current_line = current_line + 2

pG_data_dec = exp_profiles[current_line:num_data_dec + current_line] # unused
current_line = current_line + num_data_dec
no_clue = exp_profiles[current_line:current_line+2]
steady_probe_1 = exp_profiles[current_line+2]


current_line=current_line+3

probe_data_inc = exp_profiles[current_line:current_line + num_data_inc]
current_line = current_line+num_data_inc
steady_probe_2 = exp_profiles[current_line+1]

steady_probe_1_down = exp_profiles[current_line+1] # unused
current_line = current_line+2
probe_data_dec = exp_profiles[current_line:current_line+num_data_dec]# unused
current_line = current_line + num_data_dec
steady_probe_2_down = exp_profiles[current_line]
no_clue_again = exp_profiles[current_line+1]
current_line = current_line+2
time_data_inc = exp_profiles[current_line:current_line+num_data_inc]
current_line = current_line+num_data_inc
time_data_dec = exp_profiles[current_line:current_line+num_data_dec]


# modification of values into a form suitable for the model
# also adding new values for model
mO2 = mO2_raw * (273.15 + temp) * 8.314472 / 101325
mN2 = mN2_raw * (273.15 + temp) * 8.314472 / 101325
dVG = volume * gas_hold_up / (1 - gas_hold_up)
Km1 = Km1/(pi**2)
probe_dataN = (np.array(probe_data_inc) - steady_probe_2) / (steady_probe_1 - steady_probe_2)
# constants for Antoine equation for partial pressure of H20
A = 8.07131
B = -1730.63
C = 233.426
pH2O = 10 ** (A + B / (temp + C)) * 101325 / 760

print("Does the gas input in konst.dta equal xxx.dtm? " + str(gas_in2 == gas_in))
time_points = 1100
t = np.linspace(0, max(time_data_inc), num=time_points)
def probe_polynomial():
    probe_interpol=np.interp(time_data_inc[index:],time_data_inc,probe_dataN)
    # probe_data_inc_N = (np.array(probe_data_inc)-steady_probe_2)/(steady_probe_1-steady_probe_2)
    # x = time_data_inc
    # y = probe_data_inc_N
    # pol = np.polyfit(x,y,2)
    #
    # data = np.polyval(pol,t)
    # probe_data =data#(data-data.max())/(data.min()-data.max())
    # return probe_data
    return probe_interpol
def impulse_response():


    n = np.linspace(0, 1000, num=1001)

    It_Opt = np.zeros_like(t)  # create an array to store the results
    Ht_Opt = np.zeros_like(t)
    for i, ti in enumerate(t):
        It_Opt[i] = np.sum(-8 * exp(-pi ** 2 * Km1 * ti * ((2 * n + 1) ** 2) / 4) * (
                    (1 / ((2 * n + 1) ** 2 * pi ** 2)) * ((-pi ** 2 * Km1 * (2 * n + 1) ** 2) / 4)))
    for i,ti in enumerate(t):
        Ht_Opt [i] = np.sum (1-8*exp(-pi**2*Km1*ti*(2*n+1)**2/4)*(1/(2*n+1)**2*pi**2))
    Ht_Opt_N = (Ht_Opt - Ht_Opt.min()) / (Ht_Opt.max() - Ht_Opt.min())

    return Ht_Opt_N#(It_Opt - It_Opt.min()) / (It_Opt.max() - It_Opt.min())

def edit_profiles_for_convolution(model_profile):
    global num_ones
    global num_zeros
    num_ones = int(500)
    num_zeros = int(0)
    model_profile = np.concatenate((np.zeros(num_ones), model_profile))
    model_profile_edited = np.concatenate([model_profile, np.zeros(num_zeros)])
    #It_flipped = np.flip(It)
    It_edited = np.concatenate([np.zeros(num_ones), Ht])

    # plt.plot(np.linspace(0,max(t),num = 3101+500),It_edited)
    # plt.plot(np.linspace(0, max(t), num=3101+500), model_profile_edited)
    #
    # plt.show()
    return model_profile_edited,It_edited

def spline_pG():
    #TODO figure out what spline and filter to use so the data starts at 1
    def savitzky_golay_filter(data, window_size, order):
        return savgol_filter(data, window_size, order)

    xG = (np.array(pG_data_inc) - steady_pG_2) / (steady_pG_1 - steady_pG_2)
    xG = savitzky_golay_filter(xG, window_size=8, order=2)

    return CubicSpline(time_data_inc, xG)
index=np.where(probe_dataN>=upper_limit)[0].max()+1
time_data_for_compare = time_data_inc[index:]
cs =spline_pG()
Ht = impulse_response()
probe_profile = probe_polynomial()

# this line helps determine what index will be used for the start of the comparing times

#print(time_data_inc[index]), from this time the comparisons will start


def to_opt(kla):
    global der_x
    der_x = []
    p1 = steady_pG_1 + pG_atm
    p2 = steady_pG_2 + pG_atm

    print(kla)


    def dSdt(t, S):


        xO2L, xN2L, xO2G = S

        dxO2L = kla * (xO2G - xO2L)
        dxN2L = kla * (difN2 / difO2) ** (0.5) * ((cs(t) - xO2G * y) / (1 - y) - xN2L)

        gas_out = (gas_in * (p2 - pH2O) / (p1 - p2) - volume * (
                mO2 * dxO2L * y + mN2 * dxN2L * (1 - y)) - dVG * cs.derivative()(t)) / (
                                       cs(t) + (p2 - pH2O) / (p1 - p2))

        dxO2G = (gas_in * (p2 - pH2O) / (
                p1 - p2) - volume * mO2 * dxO2L - gas_out * (xO2G + (

                p2 - pH2O) / (p1 - p2))) / dVG
        #print('dxO2L =', dxO2L)
        # print('kla =', kla)
        #print('xO2G =', xO2G)
        #print('xO2L =', xO2L)
        # print('dxN2L =', dxN2L)
        # print('difN2 =', difN2)
        # print('difO2 =', difO2)
        #
        #print('t =', t)
        # print('y =', y)
        # print('xN2L =', xN2L)
        #print('gas_out =', gas_out)
        # print('gas_in =', gas_in)
        # print('p2 =', p2)
        # print('pH2O =', pH2O)
        # print('p1 =', p1)
        # print('volume =', volume)
        # print('mO2 =', mO2)
        # print('mN2 =', mN2)
        # print('dVG =', dVG)
        # print('dxO2G =', dxO2G)
        der_x.append(dxO2L[0])

        return np.array([dxO2L,
                 dxN2L,
                 dxO2G]).flatten()

    # initial conditions, they start from 1 because the profiles of x02L,x02_G,xN2L are normalized from 1 to 0.

    xO2L_0 = 1
    xO2G_0 = 1
    xN2L_0 = 1
    S_0 = (xO2L_0, xN2L_0, xO2G_0)

    sol = odeint(dSdt, y0=S_0, t=t, tfirst=True)

    O2L = sol[:, 0]
    N2L = sol[:, 1]
    O2G = sol[:, 2]
    neco = O2L,N2L,O2G
    nene=dSdt(t,neco)
    DX = nene[0:1100]

    cs_x = CubicSpline(np.linspace(0,t.max(),num=len(der_x)),der_x)
    # plt.plot(t,data)
    #plt.plot(time_for_der,der_x)
    # plt.show()
    # plt.plot(t,O2L)
    # plt.plot(t,N2L)
    # plt.plot(t,O2G)
    #
    # plt.legend(["O2L","N2L","O2G"])
    # plt.show()
    # plt.plot(t,O2L)
    # plt.plot(t,cs_x(t))
    # plt.plot(t,It)
    # plt.show()
    model_profile = cs_x(t)


    time_data_for_compare = time_data_inc[index:]
    G1 = np.zeros(len(time_data_for_compare))



    vector = np.zeros((len(t), len(time_data_for_compare)))

    for k in range(1,len(time_data_for_compare),1):

        i = np.where(t>=time_data_for_compare[k])[0].min()
        if i == 0:
            G1[k] =0
        else:
            # print(i)

            vector[0:i,k] = DX[0:i]* np.flip(Ht[0:i])
            # print(der_O2L[0:i])
            # print(np.flip(Ht[0:i]))
            #print(vector[0:i,k])

            G1[k] = simps(t[0:i],vector[0:i,k])
            #print(G1[k])
            #print("hah")


    np.set_printoptions(threshold=np.inf)
    #print(vector.view())


    probe_dataN = (np.array(probe_data_inc) - steady_probe_2) / (steady_probe_1 - steady_probe_2)

    G2=1-G1
    plt.plot(time_data_for_compare, G2,label="konvoluce")
    plt.plot(time_data_inc[index:],probe_profile,label ="Probe profile")
    plt.plot(t,O2L,label = "model profile")
    plt.plot(time_data_inc, probe_dataN,"ro",markersize=1,label = "Probe data")
    plt.legend()
    plt.show()
    # TODO jak return vic hodnot??

    return sum((G2 - probe_profile) ** 2)



def opt(choice):
    x0=0.033


    if choice == 1: # options={"maxiter":1,"disp": True}
        return scipy.optimize.minimize(to_opt, x0, method ="Nelder-Mead").x
    elif choice == 2:
        return scipy.optimize.minimize(to_opt, x0,method ="BFGS").x
    elif choice == 3:
        return scipy.optimize.minimize(to_opt, x0,method ="Powell").x
    else:
        print(choice)


opt(1)
if __name__=="__main__":
    pass