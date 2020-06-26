
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

min_power = -2
max_power = 2
point_count = 401
power_range = np.logspace(min_power, max_power, point_count)


class Precoder(object):

    def __init__(self, antenna_x: int, antenna_y: int, user_count: int, l0: int, l1: int, freq: int, itr: int):
        """initializing precoder"""
        self.__antenna_x = antenna_x
        self.__antenna_y = antenna_y
        self.__antennas = antenna_x * antenna_y
        self.__users = user_count
        self.__length_before = l0
        self.__length_after = l1
        self.__freq = freq
        self.__block = 100
        self.__prefix = 20
        self.__itr = itr
        self.__unit = 0.5
        self.__power = power_range
        return

    @staticmethod
    def __bit_generator(var: float, size: tuple) -> np.ndarray:
        """generating random stream of bits"""
        std = np.sqrt(var / 2)
        vector_rl = np.random.normal(0, std, size=size)
        vector_im = np.random.normal(0, std, size=size)
        vector = vector_rl + 1j * vector_im
        return vector

    @staticmethod
    def get_norm(matrix: np.ndarray) -> float:
        """obtaining matrix second degree norm"""
        norm = np.zeros(shape=matrix.shape[0], dtype='cfloat')
        for index in range(0, matrix.shape[0]):
            norm[index] = np.trace(matrix[index].transpose().conj() @ matrix[index])
        factor = np.sum(norm, axis=0)
        factor = np.abs(factor)
        factor = np.sqrt(factor)
        return factor

    def to_freq(self, matrix: np.ndarray) -> np.ndarray:
        """translating matrix to frequency"""
        size = (self.__freq, matrix.shape[1], matrix.shape[2])
        matrix_f = np.zeros(shape=size, dtype='cfloat')
        size = (self.__length_before, matrix.shape[1], matrix.shape[2])
        for tap_f in range(0, self.__freq):
            temp = np.zeros(shape=size, dtype='cfloat')
            for tap_t in range(0, self.__length_before):
                temp[tap_t] = np.exp(-2j * np.pi * tap_f * tap_t / self.__length_before) * matrix[tap_t]
            matrix_f[tap_f] = np.sum(temp, axis=0)
        return matrix_f

    def to_time(self, matrix: np.ndarray) -> np.ndarray:
        """translating matrix to time"""
        size = (self.__length_after, matrix.shape[1], matrix.shape[2])
        matrix_t = np.zeros(shape=size, dtype='cfloat')
        size = (self.__freq, matrix.shape[1], matrix.shape[2])
        for tap_t in range(0, self.__length_after):
            temp = np.zeros(shape=size, dtype='cfloat')
            for tap_f in range(0, self.__freq):
                temp[tap_f] = np.exp(2j * np.pi * tap_t * tap_f / self.__freq) * matrix[tap_f]
                # temp[tap_f] = np.exp(2j * np.pi * tap_t * tap_f / self.__freq) * matrix[tap_f]
            matrix_t[tap_t] = np.mean(temp, axis=0)
        return matrix_t

    def get_distance(self, mode: str) -> np.ndarray:
        """generating distance matrix"""
        distance = np.zeros(shape=(self.__antennas, self.__antennas))
        if mode == 'linear':
            for row in range(0, self.__antennas):
                for col in range(0, self.__antennas):
                    distance[row][col] = np.abs(row - col)
        elif mode == 'rectangular':
            row = 0
            for row_0 in range(0, self.__antenna_x):
                for col_0 in range(0, self.__antenna_y):
                    col = 0
                    for row_1 in range(0, self.__antenna_x):
                        for col_1 in range(0, self.__antenna_y):
                            distance[row][col] = np.sqrt(np.power(col_1 - col_0, 2) + np.power(row_1 - row_0, 2))
                            col += 1
                    row += 1
        else:
            raise Exception('mode=' + mode + ' is not valid!')
        distance *= self.__unit
        return distance

    def get_corr(self, mode: str, alpha: float) -> np.ndarray:
        """generating correlation matrix"""
        distance = self.get_distance(mode=mode)
        correlation = np.power(alpha, distance)
        return correlation

    def get_corr_fact(self, mode: str, alpha: float) -> float:
        """generating correlation matrix factor"""
        correlation = self.get_corr(mode=mode, alpha=alpha)
        factor = np.trace(correlation @ correlation.transpose().conj())
        factor = np.abs(factor)
        return factor

    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """normalizing matrix"""
        for index in range(0, self.__itr):
            norm = self.get_norm(matrix=matrix[index])
            matrix[index] = matrix[index] / norm
        return matrix

    def get_delay(self) -> np.ndarray:
        """generating power delay profile"""
        theta = np.array([p_temp / 5 for p_temp in range(0, self.__users)])
        depth_local = np.array([d_temp for d_temp in range(0, self.__length_before)])
        power_profile = []
        for delay_tap in range(0, self.__length_before):
            power_local = np.zeros(shape=self.__users)
            for user in range(0, self.__users):
                delay_temp = np.exp(-1 * theta[user] * depth_local)
                power_local[user] = np.exp(-1 * theta[user] * delay_tap) / np.sum(delay_temp)
            profile_temp = np.diag(power_local)
            power_profile.append(profile_temp)
        power_profile = np.array(power_profile)
        return power_profile

    def get_channel(self, var: float, mode: str, alpha: float) -> np.ndarray:
        """generating channel response matrix"""
        size = (self.__itr, self.__length_before, self.__users, self.__antennas)
        h_rand = self.__bit_generator(var=var, size=size)
        delay = self.get_delay()
        corr = self.get_corr(mode=mode, alpha=alpha)
        matrix_h = np.zeros(shape=h_rand.shape, dtype='cfloat')
        for itr in range(0, self.__itr):
            for tap_0 in range(0, self.__length_before):
                matrix_h[itr][tap_0] = linalg.sqrtm(delay[tap_0]) @ h_rand[itr][tap_0] @ linalg.sqrtm(corr)
        return matrix_h

    def get_precoder(self, matrix_h: np.ndarray) -> np.ndarray:
        """obtaining precoder matrix"""
        raise Exception('not implemented!')

    def __get_composite(self, matrix_h: np.ndarray, matrix_p: np.ndarray) -> np.ndarray:
        """obtaining composite matrix"""
        cap = np.min((self.__length_before, self.__length_after))
        size = (self.__itr, cap, self.__users, self.__users)
        matrix_d = np.zeros(shape=size, dtype='cfloat')
        for index in range(0, self.__itr):
            for tap in range(0, cap):
                matrix_d[index][tap] = matrix_h[index][tap] @ matrix_p[index][tap]
        matrix_d = np.mean(np.sum(matrix_d, axis=1), axis=0)
        return matrix_d

    def get_desired(self, matrix_h: np.ndarray, matrix_p: np.ndarray, matrix_s: np.ndarray) -> np.ndarray:
        """obtaining desired symbol matrix"""
        size = (self.__itr, self.__users, self.__block)
        matrix_g = np.zeros(shape=size, dtype='cfloat')
        matrix_d = self.__get_composite(matrix_h=matrix_h, matrix_p=matrix_p)
        for index in range(0, self.__itr):
            for user in range(0, self.__users):
                matrix_g[index][user] = matrix_d[user][user] * matrix_s[index][user]
        return matrix_g

    def get_input(self, matrix_s: np.ndarray, matrix_p: np.ndarray) -> np.ndarray:
        """obtaining input symbol matrix"""
        size = (self.__itr, self.__antennas, self.__block)
        matrix_x = np.zeros(shape=size, dtype='cfloat')
        for itr in range(0, self.__itr):
            p_local = matrix_p[itr]
            s_local = matrix_s[itr]
            mem = np.zeros(shape=(self.__antennas, self.__block), dtype='cfloat')
            for tap_1 in range(0, self.__length_after):
                for index in range(0, self.__block):
                    element = (index + tap_1) % self.__block
                    mem[:, index] += p_local[tap_1] @ s_local[:, element]
            matrix_x[itr] = mem
        return matrix_x

    def get_output(self, matrix_x: np.ndarray, matrix_h: np.ndarray) -> np.ndarray:
        """obtaining output symbol matrix"""
        size = (self.__itr, self.__users, self.__block)
        matrix_y = np.zeros(shape=size, dtype='cfloat')
        for itr in range(0, self.__itr):
            h_local = matrix_h[itr]
            x_local = matrix_x[itr]
            mem = np.zeros(shape=matrix_y[itr].shape, dtype='cfloat')
            for tap_0 in range(0, self.__length_before):
                for index in range(0, self.__block):
                    element = index - tap_0
                    mem[:, index] += h_local[tap_0] @ x_local[:, element]
            matrix_y[itr] = mem
        return matrix_y

    def search(self, rho_s: float, rho_h: float, mode: str, alpha: float) -> tuple:
        """search tree"""
        size = (self.__itr, self.__users, self.__block)
        matrix_s = self.__bit_generator(var=rho_s, size=size)
        matrix_h = self.get_channel(var=rho_h, mode=mode, alpha=alpha)
        matrix_p = self.get_precoder(matrix_h=matrix_h)
        matrix_x = self.get_input(matrix_s=matrix_s, matrix_p=matrix_p)
        matrix_y = self.get_output(matrix_x=matrix_x, matrix_h=matrix_h)
        matrix_g = self.get_desired(matrix_h=matrix_h, matrix_p=matrix_p, matrix_s=matrix_s)
        matrix_n = matrix_y - matrix_g
        var_s = np.var(matrix_g, axis=2)
        var_n = np.var(matrix_n, axis=2)
        return var_s, var_n

    def get_rate(self, rho_h=1.0, rho_n=1.0, mode='linear', alpha=0.0) -> np.ndarray:
        """getting bit rate"""
        var_s, var_n = self.search(rho_s=1.0, rho_h=rho_h, mode=mode, alpha=alpha)
        size = (self.__itr, self.__users, len(self.__power))
        ratio = np.zeros(shape=size)
        for index in range(0, self.__itr):
            for user in range(0, self.__users):
                ratio[index][user] = var_s[index][user] * self.__power / (var_n[index][user] * self.__power + rho_n)
        rate = 0.5 * np.log2(1 + ratio)
        rate = np.mean(np.sum(rate, axis=1), axis=0)
        return rate

    def get_max(self, rho_h=1.0, mode='linear', alpha=0.0):
        """getting max rate possible"""
        var_s, var_n = self.search(rho_s=1, rho_h=rho_h, mode=mode, alpha=alpha)
        ratio = var_s / var_n
        rate = 0.5 * np.log2(1 + ratio)
        rate = np.mean(np.sum(rate, axis=1), axis=0)
        return rate


class MatchedFilter(Precoder):

    def __init__(self, antenna_x: int, antenna_y: int, user_count: int, l0: int, l1: int, freq: int, itr: int):
        """initializing matched filter precoder"""
        super().__init__(antenna_x, antenna_y, user_count, l0, l1, freq, itr)
        self.__antennas = antenna_x * antenna_y
        self.__users = user_count
        self.__length_before = l0
        self.__length_after = l1
        self.__freq = freq
        self.__itr = itr
        return

    def get_precoder(self, matrix_h: np.ndarray) -> np.ndarray:
        """obtaining precoder matrix"""
        if self.__length_before == self.__length_after:
            size = (self.__itr, self.__length_before, self.__antennas, self.__users)
            matrix_p = np.zeros(shape=size, dtype='cfloat')
            for itr in range(0, self.__itr):
                for tap_1 in range(0, self.__length_after):
                    matrix_p[itr][tap_1] = matrix_h[itr][tap_1].transpose().conj()
        else:
            size = (self.__itr, self.__length_after, self.__antennas, self.__users)
            matrix_p = np.zeros(shape=size, dtype='cfloat')
            size = (self.__freq, self.__antennas, self.__users)
            for itr in range(0, self.__itr):
                matrix_f = np.fft.fft(matrix_h[itr], axis=0, n=self.__freq)
                temp = np.zeros(shape=size, dtype='cfloat')
                for tap_f in range(0, self.__freq):
                    temp[tap_f] = matrix_f[tap_f].transpose().conj()
                matrix_p[itr] = np.fft.ifft(temp, axis=0, n=self.__length_after)
        matrix_p = self.normalize(matrix=matrix_p)
        return matrix_p


class ZeroForcing(Precoder):

    def __init__(self, antenna_x: int, antenna_y: int, user_count: int, l0: int, l1: int, freq: int, itr: int):
        """initializing zero forcing precoder"""
        super().__init__(antenna_x, antenna_y, user_count, l0, l1, freq, itr)
        self.__antennas = antenna_x * antenna_y
        self.__users = user_count
        self.__length_after = l1
        self.__freq = freq
        self.__itr = itr
        return

    def get_precoder(self, matrix_h: np.ndarray) -> np.ndarray:
        """obtaining precoder matrix"""
        size = (self.__itr, self.__length_after, self.__antennas, self.__users)
        matrix_p = np.zeros(shape=size, dtype='cfloat')
        size = (self.__freq, self.__antennas, self.__users)
        for itr in range(0, self.__itr):
            matrix_f = np.fft.fft(matrix_h[itr], axis=0, n=self.__freq)
            temp = np.zeros(shape=size, dtype='cfloat')
            for tap_f in range(0, self.__freq):
                nominator = matrix_f[tap_f].transpose().conj()
                denominator = matrix_f[tap_f] @ matrix_f[tap_f].transpose().conj()
                denominator = np.linalg.inv(denominator)
                temp[tap_f] = nominator @ denominator
            matrix_p[itr] = np.fft.ifft(temp, axis=0, n=self.__length_after)
        matrix_p = self.normalize(matrix=matrix_p)
        return matrix_p


class RegularizedZeroForcing(Precoder):

    def __init__(self, antenna_x: int, antenna_y: int, user_count: int, l0: int, l1: int, freq: int, itr: int):
        """initializing zero forcing precoder"""
        super().__init__(antenna_x, antenna_y, user_count, l0, l1, freq, itr)
        self.__antennas = antenna_x * antenna_y
        self.__users = user_count
        self.__length_after = l1
        self.__freq = freq
        self.__itr = itr
        return

    def get_precoder(self, matrix_h: np.ndarray) -> np.ndarray:
        """obtaining precoder matrix"""
        size = (self.__itr, self.__length_after, self.__antennas, self.__users)
        matrix_p = np.zeros(shape=size, dtype='cfloat')
        size = (self.__freq, self.__antennas, self.__users)
        identity = np.ones(shape=self.__antennas)
        identity = np.diag(identity)
        for itr in range(0, self.__itr):
            matrix_f = np.fft.fft(matrix_h[itr], axis=0, n=self.__freq)
            temp = np.zeros(shape=size, dtype='cfloat')
            for tap_f in range(0, self.__freq):
                nominator = matrix_f[tap_f].transpose().conj()
                denominator = matrix_f[tap_f] @ matrix_f[tap_f].transpose().conj() + 2 * identity
                denominator = np.linalg.inv(denominator)
                temp[tap_f] = nominator @ denominator
            matrix_p[itr] = np.fft.ifft(temp, axis=0, n=self.__length_after)
        matrix_p = self.normalize(matrix=matrix_p)
        return matrix_p


class Visualizer:

    def __init__(self):
        """initializing data visualizer"""
        self.__users = 10
        self.__l0 = 4
        self.__l1 = 4
        self.__freq = 4
        self.__itr = 4
        return

    def __viz_rate(self, precoder: str, alpha: float, mode: str, color: str, marker: str):
        """drawing single rate figure"""
        rows = 16
        cols = 8
        power_db = np.array([10 * np.log10(p_temp) for p_temp in power_range])
        if precoder == 'matched':
            pre = MatchedFilter
        elif precoder == 'zero':
            pre = ZeroForcing
        else:
            pre = RegularizedZeroForcing
        model = pre(rows, cols, self.__users, self.__l0, self.__l1, self.__freq, self.__itr)
        num = model.get_rate(mode=mode, alpha=alpha)
        plt.plot(power_db, num, c=color, marker=marker, mec=color, mfc='None', ms=10)
        return

    def __viz_max_rate(self, precoder: str, alpha: float, mode: str, color: str, marker: str):
        """drawing single max rate figure"""
        rows = np.array((4, 4, 8, 8, 16, 16))
        cols = np.array((8, 16, 16, 32, 32, 64))
        size = len(rows)
        max_rate = np.zeros(shape=size)
        if precoder == 'matched':
            pre = MatchedFilter
        elif precoder == 'zero':
            pre = ZeroForcing
        else:
            pre = RegularizedZeroForcing
        for index in range(0, size):
            model = pre(rows[index], cols[index], self.__users, self.__l0, self.__l1, self.__freq, self.__itr)
            max_rate[index] = model.get_max(mode=mode, alpha=alpha)
        plt.plot(rows * cols, max_rate, c=color, marker=marker, mec=color, mfc='None', ms=10)
        return

    def viz_max_rate(self, precoder: str, mode: str):
        """drawing series of figures"""
        alpha_list = np.array((0.0, 0.5, 0.9))
        if precoder is None:
            precoder = np.array(('matched', 'zero', 'regularized'))
        else:
            precoder = np.array(precoder)
        if mode is None:
            mode = np.array(('linear', 'rectangular'))
        else:
            mode = np.array(mode)
        for pre in precoder:
            for md in mode:
                for alpha in alpha_list:
                    self.__viz_max_rate(precoder=pre, alpha=alpha, mode=md, color='', marker='')
        plt.show()
        return


if __name__ == '__main__':
    """test run"""


