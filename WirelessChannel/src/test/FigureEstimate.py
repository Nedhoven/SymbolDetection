
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


error_opt_log = [0.49544999999999995, 0.11302100000000004, 0.00150300000000001, 0.0003009999999999447]
error_app_log = [0.49496300000000004, 0.11302100000000004, 0.0026840000000000188, 0.0004110000000000502]
error_opt_bay = [0.49544999999999995, 0.11302100000000004, 0.00170300000000001, 0.0003609999999999447]
error_app_bay = [0.49496300000000004, 0.110902100000000004, 0.0031840000000000188, 0.0006110000000000502]
error_bld = [0.5, 0.45, 0.35, 0.2]
power = np.logspace(-1, 2, 4)
min_power = -1
max_power = 2
markers = 40
points = 201


def func_blind(a, b, c, d, p):
    """defining blind function"""
    y = a / (b + c * np.exp(-1 * d * p))
    return y


def func_smart(a, b, p):
    """defining smart function"""
    y = a * np.exp(-1 * b * p)
    return y


def estimate_0(f):
    """estimating function for blind scenario"""
    y = np.zeros(4)
    y[0] = error_bld[0] - func_blind(f[0], f[1], f[2], f[3], power[0])
    y[1] = error_bld[1] - func_blind(f[0], f[1], f[2], f[3], power[1])
    y[2] = error_bld[2] - func_blind(f[0], f[1], f[2], f[3], power[2])
    y[3] = error_bld[3] - func_blind(f[0], f[1], f[2], f[3], power[3])
    return y


def log_estimate_1(f):
    """estimating function for scenario 1"""
    y = np.zeros(2)
    y[0] = error_app_log[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_app_log[1] - func_smart(f[0], f[1], power[1])
    return y


def log_estimate_2(f):
    """estimating function for scenario 2"""
    y = np.zeros(2)
    y[0] = error_app_log[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_app_log[2] - func_smart(f[0], f[1], power[2])
    return y


def log_estimate_3(f):
    """estimating function for scenario 3"""
    y = np.zeros(2)
    y[0] = error_opt_log[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_opt_log[2] - func_smart(f[0], f[1], power[2])
    return y


def bay_estimate_1(f):
    """estimating function for scenario 1"""
    y = np.zeros(2)
    y[0] = error_app_bay[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_app_bay[1] - func_smart(f[0], f[1], power[1])
    return y


def bay_estimate_2(f):
    """estimating function for scenario 2"""
    y = np.zeros(2)
    y[0] = error_app_bay[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_app_bay[2] - func_smart(f[0], f[1], power[2])
    return y


def bay_estimate_3(f):
    """estimating function for scenario 3"""
    y = np.zeros(2)
    y[0] = error_opt_bay[0] - func_smart(f[0], f[1], power[0])
    y[1] = error_opt_bay[2] - func_smart(f[0], f[1], power[2])
    return y


def plotter_0() -> bool:
    """visualizer for blind scenario"""
    guess = np.array([1, 1, 1, 1])
    z = np.array(fsolve(estimate_0, guess))
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_blind(z[0], z[1], z[2], z[3], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='red', marker='*', markevery=markers, label='Blind Prediction')
    return True


def log_plotter_1() -> bool:
    """visualizer for scenario 1"""
    guess = np.array([1, 1])
    z = fsolve(log_estimate_1, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='green', marker='d', markevery=markers, label='Scenario 1')
    return True


def log_plotter_2() -> bool:
    """visualizer for scenario 2"""
    guess = np.array([1, 1])
    z = fsolve(log_estimate_2, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='purple', marker='^', markevery=markers, label='Scenario 2')
    return True


def log_plotter_3() -> bool:
    """visualizer for scenario 3"""
    guess = np.array([1, 1])
    z = fsolve(log_estimate_3, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='blue', marker='o', markevery=markers, label='Scenario 3')
    return True


def bay_plotter_1() -> bool:
    """visualizer for scenario 1"""
    guess = np.array([1, 1])
    z = fsolve(bay_estimate_1, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='green', marker='d', markevery=markers, label='Scenario 1')
    return True


def bay_plotter_2() -> bool:
    """visualizer for scenario 2"""
    guess = np.array([1, 1])
    z = fsolve(bay_estimate_2, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='purple', marker='^', markevery=markers, label='Scenario 2')
    return True


def bay_plotter_3() -> bool:
    """visualizer for scenario 3"""
    guess = np.array([1, 1])
    z = fsolve(bay_estimate_3, guess)
    x = np.logspace(min_power, max_power, points)
    y = np.array([func_smart(z[0], z[1], p) for p in x])
    x = np.array([10 * np.log10(p) for p in x])
    plt.semilogy(x, y, color='blue', marker='o', markevery=markers, label='Scenario 3')
    return True


def log_simulation() -> bool:
    """result straight out of simulations"""
    log = [0.496391, 0.376151, 0.228835, 0.1166, 0.0429583, 0.0120573, 0.0039273, 0.0014723, 0.000301, 2.34e-05]
    p0 = np.logspace(-1, 2, 10)
    p0 = np.array([10 * np.log10(temp) for temp in p0])
    plt.semilogy(p0, log, color='blue', marker='o', markevery=1, label='Scenario 3')
    return True


def bay_simulation() -> bool:
    """result straight out of simulations"""
    b1 = []
    b2 = []
    p0 = np.logspace(-1, 2, 10)
    p0 = np.array([10 * np.log10(temp) for temp in p0])
    plt.semilogy(p0, b1, color='blue', marker='o', markevery=1, label='Scenario 3')
    return True
