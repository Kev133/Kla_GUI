"""
This module contains the mathematical model and optimization function
it also loads data from the files "konstant.dta" and "xxxx.dtm"
"""

# inc and dec are used for the words increase and decrease
import os
import scipy
import numpy as np
from scipy.integrate import odeint, simps
from scipy.interpolate import CubicSpline, LSQUnivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize
pi = np.pi
exp = np.exp
kla_list = []
list_excel = []


# def opt(choice,to_opt, kla_estimate):
#     if choice == 1:  # options={"maxiter":1,"disp": True}
#         return scipy.optimize.minimize(to_opt, kla_estimate, method="Nelder-Mead", tol=1e-6).x
#     elif choice == 2:
#         return scipy.optimize.minimize(to_opt, kla_estimate, method="COBYLA").x
#     elif choice == 3:
#         return scipy.optimize.minimize(to_opt, kla_estimate, method="Powell").x
#     else:
#         print(choice)

def fix_float(value):
    """files like konstant.dta sometimes have numbers in the format e.g. 1.5d-5
     this function converts this to the standard 1.5e-5, so that Python can recognize it"""
    try:
        return float(value)
    except ValueError:  # if a float cannot be made from the string, the string is edited
        return float(value.replace("d", "e"))


def find_experiment_number(experiment_name):
    """Finds the number of the experiment from 0-17 (18 total) from the experiment name
     which is in the format e.g. PBD23C, this would equal number 8
      this number is then used to find the correct line in konstant.dta for the exp."""

    gas_in_number = int(experiment_name[3]) - 1
    impeller_frequency_num = int(experiment_name[4])
    experiment_number = 6 * gas_in_number + impeller_frequency_num - 1

    return experiment_number


def load_data(dtm_file, konstant_file, directory_name):
    """
    function for extracting data from the konstant.dta file and xxx.dtm file.
    The data in these files is seperated into variables which populate
    the dictionary model_input returned by this function.
    """

    # trying to obtain just the experiment name
    experiment_name = dtm_file.replace(directory_name, "")
    experiment_name = experiment_name.replace("\\", "")
    experiment_name = experiment_name.replace(".dtm", "")
    # loading the data from the .dtm file into exp_profiles
    with open(dtm_file, "r") as f:
        exp_profiles = f.read().splitlines()

    exp_profiles = list(map(float, exp_profiles))
    temp = exp_profiles[2]
    model_input = ({
        "experiment_name": experiment_name,
        "temp": temp,
        "pG_atm": exp_profiles[3] * 101325 / 760,
        "gas_in_flow_raw": exp_profiles[4],
        "gas_in_flow":exp_profiles[4] / 1000 / 60,
        "agitator_frequency": exp_profiles[5],
        "num_of_channels": exp_profiles[6],
        "y": exp_profiles[7],
        "measurement_date": exp_profiles[8],
        "measurement_time": exp_profiles[9],
    })
    num_data_inc = int(exp_profiles[10])
    num_data_dec = int(exp_profiles[11])
    model_input["steady_pG_1"] = exp_profiles[12]

    current_line = 13
    model_input["pG_data_inc"] = exp_profiles[current_line:num_data_inc + current_line]

    current_line = current_line + num_data_inc

    model_input["steady_pG_2"] = exp_profiles[current_line + 1]

    steady_pG_2_down = exp_profiles[current_line + 1]  # unused

    current_line = current_line + 2

    pG_data_dec = exp_profiles[current_line:num_data_dec + current_line]  # unused
    current_line = current_line + num_data_dec
    no_clue = exp_profiles[current_line:current_line + 2]
    model_input["steady_probe_1"] = exp_profiles[current_line + 2]

    current_line = current_line + 3

    model_input["probe_data_inc"] = exp_profiles[current_line:current_line + num_data_inc]
    current_line = current_line + num_data_inc
    model_input["steady_probe_2"] = exp_profiles[current_line + 1]

    steady_probe_1_down = exp_profiles[current_line + 1]  # unused
    current_line = current_line + 2
    probe_data_dec = exp_profiles[current_line:current_line + num_data_dec]  # unused
    current_line = current_line + num_data_dec
    steady_probe_2_down = exp_profiles[current_line + 1]  # unused
    no_clue_again = exp_profiles[current_line + 1]
    current_line = current_line + 2
    time_data_inc = exp_profiles[current_line:current_line + num_data_inc]
    model_input["time_data_inc"] = time_data_inc
    current_line = current_line + num_data_inc
    time_data_dec = exp_profiles[current_line:current_line + num_data_dec]  # unused

    # .DTA FILE
    # seperating data from konstant.dta into lines
    with open(konstant_file,encoding="utf-8") as f:
        konstant = f.readlines()

    model_input["header"] = konstant[0].strip()
    probe_name = konstant[1].strip()
    model_input["probe_name"] = probe_name.replace("'","")
    Km1, Km2, Zg1 = map(float, konstant[2].split())
    mO2_raw, mN2_raw, difO2, difN2 = map(fix_float, konstant[3].split())
    volume = float(konstant[4]) / 1000
    process_variables = []
    # Getting the process variables from konstant.dta
    for line in konstant[5:]:
        process_variables.append(tuple(map(float, line.split())))
    experiment_number = find_experiment_number(experiment_name)
    gas_in_flow_k, agitator_frequency_k, gas_hold_up, agitator_power = \
        process_variables[experiment_number][0:4]

    gas_in_flow_k = gas_in_flow_k / 1000 / 60

    # # constants for Antoine equation for partial pressure of H20
    A = 8.07131
    B = -1730.63
    C = 233.426
    # pH2O = 10 ** (A + B / (temp + C)) * 101325 / 760
    if not gas_in_flow_k == model_input["gas_in_flow"] and not\
            agitator_frequency_k == model_input["agitator_frequency"]:
        print("Gas input in konst.dta does equal the one in xxx.dtm ")

    # time point is how many values are there going to be in the time vector t
    time_points = 10000

    model_input.update({
        "mO2": mO2_raw * (273.15 + temp) * 8.314472 / 101325,
        "mN2": mN2_raw * (273.15 + temp) * 8.314472 / 101325,
        "difO2": difO2,
        "difN2": difN2,
        "volume": float(konstant[4]) / 1000,
        "gas_in_flow_k": gas_in_flow_k / 1000 / 60,
        "agitator_frequency_k": agitator_frequency_k,
        "gas_hold_up": gas_hold_up,
        "agitator_power": agitator_power,
        "dVG": volume * gas_hold_up / (1 - gas_hold_up),
        "Km1": Km1 / (pi ** 2),
        "probe_dataN": (np.array(model_input["probe_data_inc"]) - model_input["steady_probe_2"]) \
                       / (model_input["steady_probe_1"] - model_input["steady_probe_2"]),
        "xG": (np.array(model_input["pG_data_inc"]) - model_input["steady_pG_2"]) /
              (model_input["steady_pG_1"] - model_input["steady_pG_2"]),
        "pH2O": 10 ** (A + B / (temp + C)) * 101325 / 760,
        "t": np.linspace(0, max(time_data_inc), num=time_points),
        "p1": model_input["steady_pG_1"] + model_input["pG_atm"],
        "p2": model_input["steady_pG_2"] + model_input["pG_atm"]
    })
    unused_var_list = [Km2, Zg1, steady_probe_1_down, steady_probe_2_down, probe_data_dec,
                       pG_data_dec, time_data_dec, no_clue_again, no_clue,steady_pG_2_down]
    return model_input


def probe_characteristics(var):
    """
    Calculates the transient characteristic H(t) and the impulse characteristic It,
    both of them are normalized from 0 to 1,
    It is not used anywhere in the model, but it could be used in the future
    """
    n = np.linspace(0, 1000, num=1001)

    It_Opt = np.zeros_like(var["t"])
    Ht_Opt = np.zeros_like(var["t"])

    for i, ti in enumerate(var["t"]):
        It_Opt[i] = np.sum(-8 * exp(-pi ** 2 * var["Km1"] * ti * ((2 * n + 1) ** 2) / 4)
                           * ((1 / ((2 * n + 1) ** 2 * pi ** 2)) *
                              ((-pi ** 2 * var["Km1"] * (2 * n + 1) ** 2) / 4)))
    for i, ti in enumerate(var["t"]):
        Ht_Opt[i] = np.sum(
            1 - 8 * exp(-pi ** 2 * var["Km1"] * ti * (2 * n + 1) ** 2 / 4)
            * (1 / (2 * n + 1) ** 2 * pi ** 2))

    Ht_Opt_N = (Ht_Opt - Ht_Opt.min()) / (Ht_Opt.max() - Ht_Opt.min())
    It_Opt_N = (It_Opt - It_Opt.min()) / (It_Opt.max() - It_Opt.min())
    return Ht_Opt_N


def spline_pG(arg):
    """Noisy xG pressure time profile is fit with a spline, using the spline in the ODEs
    helps speed all the calculations by a lot"""

    # xG = savitzky_golay_filter(xG, window_size=4, order=3)
    hh = CubicSpline(arg["time_data_inc"], arg["xG"])
    knots = []
    # knots = [ 0.1,0.15,0.2,0.3]
    for i in range(1, 15):
        knots.append(arg["time_data_inc"][i * 8])
    for i in range(15, 25):
        knots.append(arg["time_data_inc"][i * 16])
    # setting up weights for spline, all are 1 instead of the first
    # the first is 10 so that the values is closest to 1 at the start
    weights_spline = np.ones(len(arg["xG"]))
    weights_spline[0] = 10

    lsq_spline = LSQUnivariateSpline(arg["time_data_inc"], arg["xG"], knots, weights_spline, ext="const")
    # der_lsq = lsq_spline.derivative()(arg["t"])
    # der_cubic = hh.derivative()(arg["t"])
    # plt.plot(arg["time_data_inc"],lsq_spline(arg["time_data_inc"]),label="lsqspline")
    # plt.plot(arg["time_data_inc"], hh(arg["time_data_inc"]),label = "cubic spline")
    # plt.plot(arg["t"],der_lsq,label="der_lsq")
    # #plt.plot(t,der_cubic,label = "der_cubic")
    # plt.legend()
    # plt.show()
    return lsq_spline  # CubicSpline(time_data_inc, xG)


def estimate_kla(var):
    """Estimating kla from simple formula , kla_estimate is used as the estimate
    in the optimize methods"""
    # print(time_data_inc[index]), from this time the comparisons will start

    index_t2 = np.where(var["probe_dataN"] >= 0.75)[0].max()
    index_t1 = np.where(var["probe_dataN"] >= 0.4)[0].max()

    kla_estimate = np.log(var["probe_dataN"][index_t2] / var["probe_dataN"][index_t1]) / (
            var["time_data_inc"][index_t1] - var["time_data_inc"][index_t2])
    # print(probe_dataN[index_t2],probe_dataN[index_t1])
    # print(time_data_inc[index_t2],time_data_inc[index_t1])

    return kla_estimate


def find_boundary_indexes(var):
    """
    This function takes the limits that were in the GUI widget "Set limits" and
    finds the indexes for these limits in probe_dataN. These are then used to
    find the time profile which will be used to compare the model and experimental values
    :param var:
    :return:
    """
    # this line helps determine what index will be used for the start of the comparing times
    try:
        index_upper = np.where(var["probe_dataN"] >= var["upper_limit"])[0].max() + 1
    except ValueError:
        index_upper = 0
    # finds the lower limit index as all the indexes that are bigger than the limit, e.g. 0.03, usually 380-400
    # this index is the end for the comparing times
    try:
        index_lower = np.where(var["probe_dataN"] <= var["lower_limit"])[0].min() + 1
    except ValueError:
        index_lower = 400
    time_data_for_compare = var["time_data_inc"][index_upper:index_lower]
    return index_lower, index_upper, time_data_for_compare


def dSdt(t, S, kla, model_params, cs):
    """
    Model equations, system of ODEs
    :param t:
    :param S:
    :param kla:
    :param model_params:
    :param cs:
    :return:
    """

    xO2L, xN2L, xO2G = S

    dxO2L = kla * (xO2G - xO2L)
    dxN2L = kla * (model_params["difN2"] / model_params["difO2"]) ** 0.5 * \
            ((cs(t) - xO2G * model_params["y"]) / (1 - model_params["y"]) - xN2L)

    gas_out = (model_params["gas_in_flow"] * (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]) - model_params["volume"] *
               (model_params["mO2"] * dxO2L * model_params["y"] + model_params["mN2"] * dxN2L * (1 - model_params["y"])) - model_params["dVG"] *
               cs.derivative()(t)) / (cs(t) + (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]))

    dxO2G = (model_params["gas_in_flow"] * (model_params["p2"] - model_params["pH2O"]) / (
            model_params["p1"] - model_params["p2"]) - model_params["volume"] * model_params["mO2"] * dxO2L - gas_out *
             (xO2G + (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]))) / model_params["dVG"]

    return np.array([dxO2L,
                     dxN2L,
                     dxO2G]).flatten()

    # initial conditions, they start from 1 because the profiles of x02L,x02_G,xN2L
    # are normalized from 1 to 0.


def calculate_ODE(kla, var, cs):
    """Calculates the oxygen,nitrogen concentrations in liquid and oxygen concentration in the gas,
    as well as their derivations. The derivations are used in the convolution integral."""
    xO2L_0 = 1
    xO2G_0 = 1
    xN2L_0 = 1
    S_0 = (xO2L_0, xN2L_0, xO2G_0)

    sol = odeint(dSdt, y0=S_0, t=var["t"], tfirst=True, args=(kla, var, cs))

    xO2L = sol[:, 0]
    xN2L = sol[:, 1]
    xO2G = sol[:, 2]
    # We have obtained the oxygen concentration profiles and
    # now are sending them back to obtain their derivates mainly dxO2L
    S_again = xO2L, xN2L, xO2G
    dxO2L = dSdt(var["t"], S_again, kla, var, cs)[0:len(var["t"])]

    # plt.plot(var["t"],xO2L)
    # plt.plot(var["t"],xN2L)
    # plt.plot(var["t"],xO2G)
    #
    # plt.legend(["O2L","N2L","O2G"])
    # plt.show()
    return xO2L, dxO2L


def convolution_integral(kla, model_params, cs, directory):
    oxygen_concentrations = calculate_ODE(kla, model_params, cs)
    xO2L = oxygen_concentrations[0]
    dxO2L = oxygen_concentrations[1]
    G1 = np.zeros(len(model_params["time_profile_for_compare"]))
    vector = np.zeros((len(model_params["t"]), len(model_params["time_profile_for_compare"])))
    # Convolution integral
    for k in range(1, len(model_params["time_profile_for_compare"]), 1):
        i = np.where(model_params["t"] >= model_params["time_profile_for_compare"][k])[0].min()

        vector[0:i, k] = dxO2L[0:i] * np.flip(model_params["Ht"][0:i])

        G1[k] = simps(vector[0:i, k], model_params["t"][0:i])

    G2 = 1+G1
    if model_params["plot_choice"] != 1:
        return sum((G2[1:] - model_params["probe_dataN"][model_params["index_upper"] + 1:model_params["index_lower"]]) ** 2)
    if model_params["plot_choice"] == 1:
        plt.clf()
        plt.plot(model_params["time_profile_for_compare"][1:], G2[1:], label="Model probe response", linewidth=0.8)
        plt.plot(model_params["time_data_inc"], model_params["xG"], label="Pressure profile", linewidth=0.8)
        # plt.plot(time_data_for_compare[1:],probe_profile[1:],label ="Probe profile",linewidth =0.8)
        # plt.plot(t,O2L,label = "model profile",linewidth = 0.8)
        plt.plot(model_params["time_data_inc"], cs(model_params["time_data_inc"]), label="Pressure spline", linewidth=0.8)
        plt.plot(model_params["time_data_inc"], model_params["probe_dataN"], "ro", markersize=0.8, label="Oxygen probe response")
        plt.title("measured by " + model_params["probe_name"] + " with constant: " + str(round(model_params["Km1"], 4)) +
            "\n kla = " + str(round(kla, 5)))
        plt.suptitle("Plot for experiment " + model_params["experiment_name"], fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory + "/Plots/Graph " + model_params["experiment_name"], dpi=700)


def minimize_functions(choice, kla_estimate, var, cs):
    if choice == 1:  # options={"maxiter":1,"disp": True}
        return minimize(convolution_integral, kla_estimate,
                                       method="Nelder-Mead", tol=1e-6, args=(var, cs,"")).x
    elif choice == 2:
        return minimize(convolution_integral, kla_estimate,
                                       method="COBYLA", tol=1e-6, args=(var, cs,"")).x
    elif choice == 3:
        return minimize(convolution_integral, kla_estimate,  #L-BFGS-B, TNC are contenders
                                       method="Powell", tol=1e-6, args=(var, cs,"")).x
def plot_results(directory,var,cs,kla):
    if not os.path.exists(directory + "/Plots"):
        os.makedirs(directory + "/Plots")
    var.update({"plot_choice": 1})
    convolution_integral(kla, var, cs,directory)

def main_function(dtm_file, konstant_file, directory, opt_method, plot_info, limits):
    """
    All the events that happen in this module
    are called from this function, this is also the function that is called by the
    GUI.py module
    """
    model_input = load_data(dtm_file, konstant_file, directory)
    model_input.update({"upper_limit": limits[0], "lower_limit": limits[1]})
    Ht = probe_characteristics(model_input)
    cs = spline_pG(model_input)
    kla_estimation = estimate_kla(model_input)
    indexes = find_boundary_indexes(model_input)
    model_input.update({"index_lower": indexes[0],
                        "index_upper": indexes[1],
                        "time_profile_for_compare": indexes[2],
                        "Ht": Ht,
                        "plot_choice": 0
                        })

    kla = minimize_functions(opt_method, kla_estimation, model_input, cs)[0]
    if plot_info == 1:
        plot_results(directory,model_input,cs,kla)
    return kla, model_input["experiment_name"], model_input["header"],model_input["gas_in_flow_raw"],\
           model_input["agitator_frequency"],model_input["gas_hold_up"]

if __name__ == "__main__":
    pass
