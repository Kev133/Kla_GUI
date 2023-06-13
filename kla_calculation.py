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
from scipy.optimize import minimize,LinearConstraint
pi = np.pi
exp = np.exp
kla_list = []
list_excel = []


def fix_float(value):
    """
    files like konstant.dta sometimes have numbers in the format e.g. 1.5d-5
     this function converts this to the standard 1.5e-5, so that Python can recognize it
     as a float.

    :param value: string of the number in the wrong format, e.g. 1.5d-5
    :return: float of the number in the correct format e.g. 1.5e-5
    """
    try:
        return float(value)
    except ValueError:  # if a float cannot be made from the string, the string is edited
        return float(value.replace("d", "e"))


def find_experiment_number(experiment_name):
    """
    Finds the number of the experiment from 0-17 (18 total) from the experiment name
    which is in the format e.g. PBD23C, this would equal number 8
    this number is then used to find the correct line in konstant.dta for the experiment.
    The first number in the example "23" is called gas_in_num as this number changes only when the inlet gas
    flow changes. The second number is the impeller frequency number, which changes with each experiment.

    :param experiment_name: a string with the number of the last two numbers from the experiment name, e.g. "23"
    :return: returns the number from the range of 0-17, if "23" was put in this function, it would return int 8.
    """

    gas_in_num = int(experiment_name[3]) - 1
    impeller_frequency_num = int(experiment_name[4])
    experiment_number = 6 * gas_in_num + impeller_frequency_num - 1

    return experiment_number


def load_data(dtm_file_dir, konstant_file_dir, dtm_files_folder_dir):
    """
    function for extracting data from the konstant.dta file and the xxx.dtm file.
    The data in these files is seperated into variables which populate
    the dictionary model_input returned by this function, some of this  data is slightly modified
    e.g. divided by 1000 so the units match later on in the model.
    Some variables are added to the dictionary which cannot be found in the .dtm or .dta files.
    These variables are :
    experiment_name: string of the name of the xxxxx.dtm file, in this case xxxxx is the name.
    pH2O: float, the vapor pressure of water in the inlet gas, calculated from the Antoine equation
    time_points: int, how many values representing time will there be from 0 to max(time_data_inc)
    correct_data_check: bool, True if the data for agitator_frequency and gas_in_flow are
    the same for the experiment. False if one of these does not reflect the other, in this case a signal with
    info that the current experiment being evaluated is wrong.

    Dictionary explanation:
    steady_Pg_1: float, the steady state pressure at the beginning of the experiments
    (no pressure change yet) , usually between 5 and 20 Pa.
    steady_Pg_2: float,the steady state pressure after the pressure change, when 400 values have been measured.
                This value is for the step change increase of pressure.

    :param dtm_file_dir: a string of the directory to the specific experimnetal .dtm file
    :param konstant_file_dir: a string of the directory of the konstant.DTA file
    :param dtm_files_folder_dir: a directory of the folder, with both these files,
                                 this string is used to get just the experiment name from the directory string
    :return: dictionary with the majority of parameters needed for evaluation.
    """

    # trying to obtain just the experiment name
    experiment_name = dtm_file_dir.replace(dtm_files_folder_dir, "")
    experiment_name = experiment_name.replace("\\", "")
    experiment_name = experiment_name.replace(".dtm", "")
    # loading the data from the .dtm file into exp_profiles

    with open(dtm_file_dir, "r") as f:
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

    with open(konstant_file_dir,encoding="utf-8") as f:
        konstant = f.readlines()

    model_input["header"] = konstant[0].strip()
    probe_name = konstant[1].strip()
    model_input["probe_name"] = probe_name.replace("'","")
    Km1, Km2, Zg1 = map(float, konstant[2].split())
    mO2_standard, mN2_standard, difO2, difN2 = map(fix_float, konstant[3].split())
    volume = float(konstant[4]) / 1000
    process_variables = []

    # Getting the process parameters from konstant.dta
    for line in konstant[5:]:
        process_variables.append(tuple(map(float, line.split())))
    experiment_number = find_experiment_number(experiment_name)
    gas_in_flow_k, agitator_frequency_k, gas_hold_up, agitator_power = \
        process_variables[experiment_number][0:4]

    gas_in_flow_k = gas_in_flow_k / 1000 / 60

    # Check if the gas flow and agitator frequency in .DTM and .DTA files match
    model_input["correct_data_check"] = True
    if not gas_in_flow_k == model_input["gas_in_flow"] or not\
            agitator_frequency_k == model_input["agitator_frequency"]:
        model_input["correct_data_check"] = False

    # time point is how many values are there going to be in the time vector t
    time_points = 10000
    # constants for Antoine equation for vapor pressure of H20
    A = 8.07131
    B = -1730.63
    C = 233.426
    model_input.update({
        "mO2": mO2_standard * (273.15 + temp) * 8.314472 / 101325,
        "mN2": mN2_standard * (273.15 + temp) * 8.314472 / 101325,
        "difO2": difO2,
        "difN2": difN2,
        "volume": float(konstant[4]) / 1000,
        "gas_in_flow_k": gas_in_flow_k / 1000 / 60,
        "agitator_frequency_k": agitator_frequency_k,
        "gas_hold_up": gas_hold_up,
        "agitator_power": agitator_power,
        "dVG": volume * gas_hold_up / (1 - gas_hold_up),
        "Km1_for_plot": Km1,
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


def probe_characteristics(model_params):
    """
    Calculates the transient characteristic H(t) and the impulse characteristic It(t),
    both of them are normalized from 0 to 1.
    It_Opt is not used anywhere in the model, but it could be used in the future

    :param model_params: dictionary with model parameters, here "t", "Km1" are used.
    "t" is the time vector and Km1 (if multiplied by pi^2) is the time constant of the optical probe
    :return: ndarray, function values for the transient characterstic, normalized from 0 to 1
    """
    # n should be until infinity, but 1000 is in the old Matlab program, it seems to be enough
    n = np.linspace(0, 1000, num=1001)

    It_Opt = np.zeros_like(model_params["t"])
    Ht_Opt = np.zeros_like(model_params["t"])

    for i, ti in enumerate(model_params["t"]):
        It_Opt[i] = np.sum(-8 * exp(-pi ** 2 * model_params["Km1"] * ti * ((2 * n + 1) ** 2) / 4)
                           * ((1 / ((2 * n + 1) ** 2 * pi ** 2)) *
                              ((-pi ** 2 * model_params["Km1"] * (2 * n + 1) ** 2) / 4)))
    for i, ti in enumerate(model_params["t"]):
        Ht_Opt[i] = np.sum(
            1 - 8 * exp(-pi ** 2 * model_params["Km1"] * ti * (2 * n + 1) ** 2 / 4)
            * (1 / (2 * n + 1) ** 2 * pi ** 2))

    Ht_Opt_N = (Ht_Opt - Ht_Opt.min()) / (Ht_Opt.max() - Ht_Opt.min())
    It_Opt_N = (It_Opt - It_Opt.min()) / (It_Opt.max() - It_Opt.min())
    return Ht_Opt_N


def spline_pG(model_params):
    """
    Noisy xG (total pressure difference time profile normalized from 1 to 0) is fit with a cubic spline.
     There is a lot of commented lines in this function. A comment above always explains what the command below does.
     Try them out to see how big of a role the pressure profile plays for large kla values

    :param model_params: dictionary with model parameters, here "xG" and "time_data_inc" are used.
     xG: the normalized (from 1 to 0) pressure profile for the step change, for pressure increase
     time_data_inc: list of the values of the time profile
    :return: returns the smoothing spline which approximates the pressure data in the .dtm file
    """
    knots = []
    # filter the normalized pressure data

    # xG = savitzky_golay_filter(xG, window_size=4, order=3)

    # use an ordinary cubic spline, it will go through each point though, better combine with the filter
    # also you have to change the return statement to return "cubic_spline"

    # cubic_spline = CubicSpline(arg["time_data_inc"], arg["xG"])

    # add knots for the first few values of the pressure profile, this tackles the issue of
    # multiple zeros at the start of the pressure profile, the derivatives of the pressure profile
    # used later in the model are not that clean when these knots are employed though

    # knots = [ 0.05,0.1,0.15,0.2,0.3]

    # adding every 8th value for the first third part of the profile as a knot
    # for the rest a knot every 16th is put in the "knots" list
    for i in range(1, 15):
        knots.append(model_params["time_data_inc"][i * 8])
    for i in range(15, 25):
        knots.append(model_params["time_data_inc"][i * 16])

    # setting up weights for spline, all are 1 instead of the first
    # the first is 10 so that the values is closest to 1 at the start
    weights_spline = np.ones(len(model_params["xG"]))
    weights_spline[0] = 10
    # a cubic spline which is calculated using least - squares, more info at
    # scipy.interpolate.LSQUnivariateSpline
    lsq_spline = LSQUnivariateSpline(model_params["time_data_inc"], model_params["xG"], knots, weights_spline, ext="const")

    return lsq_spline


def estimate_kla(model_params):
    """
    Estimating kla from simple formula , kla_estimate is used as the estimate
    in the optimize methods
    :param model_params: dictionary with model parameters, here "probe_dataN" and "time_data_inc" are used
    probe_dataN: the normalized data measured by the oxygen probe, normalized from 1 to 0, ndarray
    time_data_inc: list of the values of the time profile
    :return: returns a float which is the initial approximation of kla
    """
    # print(time_data_inc[index]), from this time the comparisons will start
    index_t2 = np.where(model_params["probe_dataN"] >= 0.75)[0].max()
    index_t1 = np.where(model_params["probe_dataN"] >= 0.4)[0].max()

    kla_estimate = np.log(model_params["probe_dataN"][index_t2] / model_params["probe_dataN"][index_t1]) / (
            model_params["time_data_inc"][index_t1] - model_params["time_data_inc"][index_t2])
    # print(probe_dataN[index_t2],probe_dataN[index_t1])
    # print(time_data_inc[index_t2],time_data_inc[index_t1])

    return kla_estimate


def find_boundary_indexes(model_params):
    """
    This function takes the limits that were in the GUI widget "Set limits" and
    finds the indexes for these limits in probe_dataN. These are then used to
    find the time profile which will be used to compare the model and experimental values
    :param model_params: dictionary with model parameters, here "probe_dataN",
    "upper_limit", "lower_limit" and "time_data_inc" are used.
    probe_dataN: the normalized data measured by the oxygen probe, normalized from 1 to 0, ndarray
    time_data_inc: list of the values of the time profile
    lower_limit = a float of the lower limit of the interval where the probe response and model probe response
    are compared
    upper_limit = a float of the upper limit of the interval where the probe response and model probe response
    are compared
    :return: returns a tuple of the lower index used to find where the time profile for should start, upper index
    where the time profile for convolution should end and the time profile for convolution itself
    """
    # this line helps determine what index will be used for the start of the comparing times
    try:
        index_upper = np.where(model_params["probe_dataN"] >= model_params["upper_limit"])[0].max() + 1
    except ValueError:
        index_upper = 0
    # finds the lower limit index as all the indexes that are bigger than the limit, e.g. 0.03, usually 380-400
    # this index is the end for the comparing times
    try:
        index_lower = np.where(model_params["probe_dataN"] <= model_params["lower_limit"])[0].min() + 1
    except ValueError:
        index_lower = 400
    time_profile_for_convolution = model_params["time_data_inc"][index_upper:index_lower]
    return index_lower, index_upper, time_profile_for_convolution


def dSdt(t, S, kla, model_params, lsq_spline):
    """
    Model equations, system of ODEs
    :param t: time profile used for the evaluation, the step is much smaller than the time profile in .dtm file
    :param S: list of initial conditions for xO2L,xN2L,xO2G. S can also be a list with the already calculated
    normalized concentrations xO2L,xN2L,xO2G (this happens when calling the function again
    to obtain the derivates dxO2L,dxN2L, dxO2G)
    :param kla: float, initial guess of kla, after following iterations, is changes by the optimization function
    until a good estimation is found
    :param lsq_spline: smoothing spline which approximates pressure data
    gas_out_flow:
    :param model_params: dictionary with model parameters, a brief description of the ones used from the dictionary
    is included, all of them are floats
    difN2: diffusivity of nitrogen
    difO2: diffusivity of oxygen
    gas_in_flow: volumetric flow of inlet gas
    y: molar fraction of oxygen in inlet gas
    p1: steady state total pressure in the vessel at the start of the experiment
    p2: steady state total pressure after the pressure step change at the end of the experiment, for
    this program, the pressure increase is only used in the evaluation, so the end of the experiment is after
    the system reaches steady state after the pressure increase

    pH2O: vapor pressure of water in the inlet gas, [Pa]
    volume: liquid volume of batch
    mO2: solubility of nitrogen
    mN2: solubility of oxygen
    dVG: volume of gas in the liquid phase
    :return: returns a nparray with the normalized concentrations, not the derivatives!!!!, even though
    it may seem like it in the return statement, the derivates are returned after calling the function again with the
    normalized concentrations that were returned the first time. This can be seen in the following code that is used
    later in the program

    xO2L = ode_solution[:, 0]
    xN2L = ode_solution[:, 1]
    xO2G = ode_solution[:, 2]
    # We have obtained the oxygen concentration profiles and
    # now are sending them back to obtain their derivates mainly dxO2L
    S_again = xO2L, xN2L, xO2G
    dxO2L = dSdt(model_params["t"], S_again, kla, model_params, lsq_spline)[0:len(model_params["t"])]

    """

    xO2L, xN2L, xO2G = S

    dxO2L = kla * (xO2G - xO2L)
    dxN2L = kla * (model_params["difN2"] / model_params["difO2"]) ** 0.5 * \
            ((lsq_spline(t) - xO2G * model_params["y"]) / (1 - model_params["y"]) - xN2L)

    gas_out_flow = (model_params["gas_in_flow"] * (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]) - model_params["volume"] *
               (model_params["mO2"] * dxO2L * model_params["y"] + model_params["mN2"] * dxN2L * (1 - model_params["y"])) - model_params["dVG"] *
               lsq_spline.derivative()(t)) / (lsq_spline(t) + (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]))

    dxO2G = (model_params["gas_in_flow"] * (model_params["p2"] - model_params["pH2O"]) / (
            model_params["p1"] - model_params["p2"]) - model_params["volume"] * model_params["mO2"] * dxO2L - gas_out_flow *
             (xO2G + (model_params["p2"] - model_params["pH2O"]) / (model_params["p1"] - model_params["p2"]))) / model_params["dVG"]

    return np.array([dxO2L,
                     dxN2L,
                     dxO2G]).flatten()

    # initial conditions, they start from 1 because the profiles of x02L,x02_G,xN2L
    # are normalized from 1 to 0.


def calculate_ODE(kla, model_params, lsq_spline):
    """
    Calculates the oxygen,nitrogen concentrations in liquid and oxygen concentration in the gas,
    as well as their derivatives. The derivatives are used in the convolution integral.
    :param kla: float, kla value
    :param model_params: dictionary with model parameters, here "t" is used as the time profile for
    the evaluation
    :param lsq_spline: smoothing spline which approximates pressure data
    :return: returns a tuple with the theoretical normalized concentration of oxygen in the liquid xO2L and the derivative
    of this dxO2L
    """
    xO2L_0 = 1
    xO2G_0 = 1
    xN2L_0 = 1
    S_0 = (xO2L_0, xN2L_0, xO2G_0)

    ode_solution = odeint(dSdt, y0=S_0, t=model_params["t"], tfirst=True, args=(kla, model_params, lsq_spline))

    xO2L = ode_solution[:, 0]
    xN2L = ode_solution[:, 1]
    xO2G = ode_solution[:, 2]
    # We have obtained the oxygen concentration profiles and
    # now are sending them back to obtain their derivates mainly dxO2L
    S_again = xO2L, xN2L, xO2G
    dxO2L = dSdt(model_params["t"], S_again, kla, model_params, lsq_spline)[0:len(model_params["t"])]

    return xO2L, dxO2L


def convolution_integral(kla, model_params, lsq_spline, directory):
    """
    The function for the convolution integral, here the convolution of dxO2L with the transient characteristic
    Ht is calculated, in order to obtain the model probe response G, which is then compared to the real probe response
    and the sum of squares of these two arrays is returned. If the parameter in model parameters "plot choice"
    is set to 1, this function save the plots of the oxygen probe response, model probe response, pressure profile and
    spline approximation of the pressure profile.
    :param kla: float, kla value
    :param model_params: dictionary with model parameters
    :param lsq_spline: smoothing spline which approximates pressure data
    :param directory: string,  the directory in which the plot for the experiment is saved
    :return: returns a float of the sum of squares of the model probe response G and the real probe response for the
    time profile for convolution.
    """
    oxygen_concentrations = calculate_ODE(kla, model_params, lsq_spline)
    xO2L = oxygen_concentrations[0]
    dxO2L = oxygen_concentrations[1]
    G1 = np.zeros(len(model_params["time_profile_for_convolution"]))
    vector = np.zeros((len(model_params["t"]), len(model_params["time_profile_for_convolution"])))
    # Convolution integral
    for k in range(1, len(model_params["time_profile_for_convolution"]), 1):
        i = np.where(model_params["t"] >= model_params["time_profile_for_convolution"][k])[0].min()

        vector[0:i, k] = dxO2L[0:i] * np.flip(model_params["Ht"][0:i])

        G1[k] = simps(vector[0:i, k], model_params["t"][0:i])

    G2 = 1+G1

    if model_params["plot_choice"] != 1:
        return sum((G2[1:] - model_params["probe_dataN"][model_params["index_upper"] + 1:model_params["index_lower"]]) ** 2)
    # save plots for the current experiments
    if model_params["plot_choice"] == 1:
        plt.clf()
        plt.plot(model_params["time_data_inc"], model_params["probe_dataN"], "ro", markersize=0.8,
                 label="Oxygen probe response")
        plt.plot(model_params["time_profile_for_convolution"][1:], G2[1:], label="Model probe response", linewidth=0.8)
        plt.plot(model_params["time_data_inc"], model_params["xG"], label="Pressure profile", linewidth=0.8)
        plt.plot(model_params["time_data_inc"], lsq_spline(model_params["time_data_inc"]),
                 label="Spline fit of pressure profile", linewidth=0.8)
        plt.title("measured by " + model_params["probe_name"] + " with the time constant " +
                  str(round(model_params["Km1_for_plot"], 4)) +" s" + "\n $k_La$ = " + str(round(kla, 5))+" s$^{-1}$")
        plt.suptitle("Plot for experiment " + model_params["experiment_name"], fontsize=14)
        plt.legend()
        plt.xlabel("time [s]",fontsize = 10)
        plt.ylabel("normalized concentration",fontsize = 10)
        plt.tight_layout()
        plt.savefig(directory + "/Plots/Plot " + model_params["experiment_name"], dpi=700)


def minimize_functions(choice, kla_estimate, model_params, lsq_spline):
    """
    This function calls the convolution_integral function and starts the optimization, using the algorithm chosen
    by the user, default is Nelder-Mead. The convolution_integral function returns the sum of squares back
    and kla is adjusted and the model is calculated again for the new kla. The sum of squares is evaluated
    and kla is adjusted again. This repeats until a kla is found for the set tolerance of 1e-6.
    Interesting fact: None of the other minimize algorithms in scipy.optimize.minimize work, they
    maybe produce a kla value, even quickly, but its error can be e.g. 200%.
    :param choice, integer representing the choice of the optimization method, where
    1 = Nelder-Mead
    2 = COBYLA  (the constraints 0 and 5 are used for kla value)
    3 = Powell:
    :param kla_estimate: float of the initial approximation of kla
    :param model_params: dictionary with model parameters
    :param lsq_spline: smoothing spline which approximates pressure data
    :return: float, returns the optimized value of kla
    """
    if choice == 1:  # options={"maxiter":1,"disp": True}
        return minimize(convolution_integral, kla_estimate,
                        method="Nelder-Mead", tol=1e-6, args=(model_params, lsq_spline, "")).x[0]
    elif choice == 2:

        linear_constraint = LinearConstraint([[1]], [0], [5])
        return minimize(convolution_integral, kla_estimate,
                        method="COBYLA", tol=1e-6,constraints=[linear_constraint], args=(model_params, lsq_spline, "")).x[0]
    elif choice == 3:
        return minimize(convolution_integral, kla_estimate,  #L-BFGS-B, TNC are contenders
                        method="Powell", tol=1e-6, args=(model_params, lsq_spline, "")).x[0]
def plot_results(directory, model_params, lsq_spline, kla):
    """
    Creates a folder named Plots in the experiment directory, and calls the convolution_integral
    function with the optimized value of kla and with the key plot_choice set to 1, which will
    result in a plot for the experiment being created in the Plots folder.

    :param directory: string of the directory where the .dtm and .dta files are
    :param model_params: dictionary with model parameters
    :param lsq_spline: smoothing spline which approximates pressure data
    :param kla: float, optimized value of kla
    """
    if not os.path.exists(directory + "/Plots"):
        os.makedirs(directory + "/Plots")
    model_params.update({"plot_choice": 1})
    convolution_integral(kla, model_params, lsq_spline, directory)

def main_function(dtm_file, dta_file, folder_directory, opt_method, plot_info, limits):
    """
    All the events that happen in this module
    are called from this function, this is also the function that is called by the
    GUI.py module
    :param dtm_file: string, the directory of the .dtm file
    :param dta_file: string, the directory of the .dta file
    :param folder_directory: directory of the folder where the .dta and .dtm files are
    :param opt_method: integer, based on this number, the optimization algorithm is chosen
    :param plot_info: boolean, if set to True, the experiment plots will be saved
    :param limits: list of the limits set by the user, default is [0.99,0.03]
    :return: returns a list of values that can be seen in the return statement
    """

    model_input = load_data(dtm_file, dta_file, folder_directory)
    model_input.update({"upper_limit": limits[0], "lower_limit": limits[1]})
    Ht = probe_characteristics(model_input)
    lsq_spline = spline_pG(model_input)
    kla_estimation = estimate_kla(model_input)
    indexes = find_boundary_indexes(model_input)
    model_input.update({"index_lower": indexes[0],
                        "index_upper": indexes[1],
                        "time_profile_for_convolution": indexes[2],
                        "Ht": Ht,
                        "plot_choice": 0
                        })

    kla = minimize_functions(opt_method, kla_estimation, model_input, lsq_spline)
    if plot_info == 1:
        plot_results(folder_directory, model_input, lsq_spline, kla)
    return kla, model_input["experiment_name"], model_input["header"],model_input["gas_in_flow_raw"],\
           model_input["agitator_frequency"],model_input["gas_hold_up"],model_input["correct_data_check"]

if __name__ == "__main__":
    pass
