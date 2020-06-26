
import numpy as np


q_line = 0.0
s_line = (-0.5, 0, 0.5)


def get_symbols(mode: int, user: int, block: int, signal_var=1.0):
    """generating random symbols"""
    std = np.sqrt(signal_var)
    sym = np.random.uniform(-1, 1, size=(2, user, block))
    if mode == 4:
        sym = np.where(sym >= q_line, 1, sym)
        sym = np.where(sym < q_line, -1, sym)
        sym = np.asarray(np.reshape(std * sym, (2, user, block)))
    return


def get_noise(user: int, block: int, noise_var: float):
    """generating white additive Gaussian noise"""
    std = np.sqrt(noise_var / 2)
    rl = np.random.normal(0, std, size=(user, block))
    im = np.random.normal(0, std, size=(user, block))
    noise = np.array(1j * im + rl)
    return noise


def get_channel_response(user: int, antenna: int, res_var: float):
    """generating channel response model"""
    std = np.sqrt(res_var / 2)
    rl = np.random.normal(0, std, size=(user, antenna))
    im = np.random.normal(0, std, size=(user, antenna))
    res = np.array(1j * im + rl)
    return res


def get_precoder(res: np.ndarray):
    """generating channel matched filter"""
    return res.transpose().conj()
