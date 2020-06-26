
import numpy as np


class MatchedFilter:

    def __init__(self, antenna_count: int, user_count: int, channel_length: int):
        """initializing the matched filter precoder"""
        self.__expected_matrix = []
        self.__expected_factor = []
        self.__count = 0
        self.__antenna_count = antenna_count
        self.__user_count = user_count
        self.__length = channel_length
        return

    def __normalization(self, matrix: np.ndarray) -> float:
        """matrix normalizer"""
        tr = []
        for tap in range(0, self.__length):
            tr.append(np.trace(matrix[tap].transpose().conj() @ matrix[tap]))
        tr = np.abs(np.sum(tr, axis=0))
        tr = float(np.sqrt(tr))
        return tr

    def get_response(self, res: np.ndarray) -> np.ndarray:
        """matched filter response"""
        precoder = []
        for tap in range(0, self.__length):
            precoder.append(res[tap].transpose().conj())
        precoder = np.asarray(precoder)
        tr = self.__normalization(matrix=precoder)
        precoder /= tr
        precoder = np.asarray(precoder)
        return precoder

    def get_output(self, res: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """applying the matched filter precoder to the symbols"""
        precoder = self.get_response(res=res)
        block_length = len(sym)
        vector = []
        for time in range(0, block_length):
            temp = []
            for tap in range(0, self.__length):
                index = (time + tap) % block_length
                temp.append(precoder[tap] @ sym[index])
            temp = np.asarray(temp)
            temp = np.sum(temp, axis=0)
            vector.append(temp)
        vector = np.asarray(vector)
        self.__desired_instances(res=res, pre=precoder, sym=sym)
        return vector

    def __desired_instances(self, res: np.ndarray, pre: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """one instance of the desired symbols"""
        temp = []
        block_length = len(sym)
        for tap in range(0, self.__length):
            temp.append(res[tap] @ pre[tap])
        matrix = np.sum(temp, axis=0)
        instance = matrix
        self.__expected_factor.append(instance)
        desired = []
        for time in range(0, block_length):
            desired.append(matrix @ sym[time])
        desired = np.asarray(desired)
        self.__load_desired(desired)
        return desired

    def __load_desired(self, desired: np.ndarray):
        """desired symbols over time"""
        self.__expected_matrix.append(desired)
        self.__count += 1
        return

    def get_expected(self) -> np.ndarray:
        """getting method for expected matrix"""
        matrix = np.asarray(self.__expected_matrix)
        return matrix

    def get_instance(self) -> np.ndarray:
        """getting method for expected matrix factor"""
        matrix = np.asarray(self.__expected_factor)
        return matrix

    def get_gain(self) -> float:
        """desired signal power in theory"""
        signal = float(self.__antenna_count / self.__user_count)
        return signal

    def get_rate(self, factor: float, power: np.ndarray) -> np.ndarray:
        """achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * power
        noise = (factor / self.__antenna_count) * power + 1
        ratio = abs(signal / noise)
        rate = [(self.__user_count / 2) * np.log2(1 + element) for element in ratio]
        rate = np.asarray(rate)
        return rate

    def get_power(self, factor: float, rate: float) -> float:
        """required power in theory"""
        r = 2 * rate / self.__user_count
        temp = (self.__antenna_count * self.__user_count * (2 ** r - 1)) /\
               (self.__antenna_count * self.__antenna_count - self.__user_count * factor * (2 ** r - 1))
        power = 10 * np.log10(abs(temp))
        return power

    def get_limit(self, factor: float, max_power: float) -> float:
        """maximum achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * max_power
        noise = (factor / self.__antenna_count) * max_power + 1
        snr = abs(signal / noise)
        cap = (self.__user_count / 2) * np.log2(1 + snr)
        return cap


class ZeroForcing:

    def __int__(self, response: np.ndarray):
        """initializing the zero forcing precoder"""
        self.__expected_matrix = []
        self.__expected_instance = []
        self.__count = 0
        self.__response = response
        self.__antenna_count = len(response[0])
        self.__user_count = len(response[0][0])
        self.__length = len(response)
        return

    def __normalization(self, matrix) -> float:
        """matrix normalizer"""
        tr = []
        for tap in range(0, self.__length):
            tr.append(np.trace(matrix[tap].transpose().conj() @ matrix[tap]))
        tr = np.abs(np.sum(tr, axis=0))
        tr = float(np.sqrt(tr))
        return tr

    def get_response(self) -> np.ndarray:
        """the channel matched filter response"""
        freq_response = np.fft.fft(self.__response, axis=0)
        freq_length = len(freq_response)
        precoder = []
        for v in range(0, freq_length):
            factor = np.linalg.inv(freq_response[v] @ freq_response[v].conj().transpose())
            precoder.append(freq_response[v].transpose().conj() @ factor)
        precoder = np.asarray(precoder)
        precoder = np.fft.ifft(precoder, axis=0)
        tr = self.__normalization(precoder)
        precoder /= tr
        precoder = np.asarray(precoder)
        return precoder

    def get_output(self, sym: np.ndarray) -> np.ndarray:
        """applying the zero forcing to the symbols"""
        block_length = len(sym)
        precoder = self.get_response()
        vector = []
        for t in range(0, block_length):
            temp = []
            for tap in range(0, self.__length):
                index = (t + tap) % block_length
                temp.append(precoder[tap] @ sym[index])
            temp = np.asarray(temp)
            temp = np.sum(temp, axis=0)
            vector.append(temp)
        vector = np.asarray(vector)
        self.__desired_instance(precoder=precoder, sym=sym)
        return vector

    def __desired_instance(self, precoder: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """one instance of the desired symbols"""
        block_length = len(sym)
        temp = []
        for tap in range(0, self.__length):
            temp.append(self.__response[tap] @ precoder[tap])
        matrix = np.sum(temp, axis=0)
        instance = np.diag(matrix)
        self.__expected_instance.append(instance)
        desired = []
        for time in range(0, block_length):
            desired.append(matrix @ sym[time])
        desired = np.asarray(desired)
        self.__load_desired(desired=desired)
        return desired

    def __load_desired(self, desired: np.ndarray):
        """desired symbols over time"""
        self.__expected_matrix.append(desired)
        self.__count += 1
        return

    def get_expected(self) -> np.ndarray:
        """getting method for expected matrix"""
        return np.asarray(self.__expected_matrix)

    def get_instance(self) -> np.ndarray:
        """getting method for expected instance"""
        return np.asarray(self.__expected_instance)

    def get_gain(self) -> float:
        """desired signal power in theory"""
        signal = float(self.__antenna_count / self.__user_count)
        return signal

    def get_rate(self, factor: float, power: np.ndarray) -> np.ndarray:
        """achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * power
        noise = (factor / self.__antenna_count) * power + 1
        ratio = abs(signal / noise)
        rate = [(self.__user_count / 2) * np.log2(1 + element) for element in ratio]
        rate = np.asarray(rate)
        return rate

    def get_power(self, factor: float, rate: float) -> float:
        """required power in theory"""
        r = 2 * rate / self.__user_count
        temp = (self.__antenna_count * self.__user_count * (2 ** r - 1)) /\
               (self.__antenna_count * self.__antenna_count - self.__user_count * factor * (2 ** r - 1))
        power = 10 * np.log10(abs(temp))
        return power

    def get_limit(self, factor: float, max_power: float) -> float:
        """maximum achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * max_power
        noise = (factor / self.__antenna_count) * max_power + 1
        snr = abs(signal / noise)
        cap = (self.__user_count / 2) * np.log2(1 + snr)
        return cap


class RegularizedZeroForcing:

    def __int__(self, response: np.ndarray, beta=2.00):
        """initializing the regularized zero forcing precoder"""
        self.__expected_matrix = []
        self.__expected_instance = []
        self.__count = 0
        self.__response = response
        self.__antenna_count = len(response[0])
        self.__user_count = len(response[0][0])
        self.__length = len(response)
        self.__beta = beta
        return

    def __normalization(self, matrix) -> float:
        """matrix normalizer"""
        tr = []
        for tap in range(0, self.__length):
            tr.append(np.trace(matrix[tap].transpose().conj() @ matrix[tap]))
        tr = np.abs(np.sum(tr, axis=0))
        tr = float(np.sqrt(tr))
        return tr

    def get_response(self) -> np.ndarray:
        """regularized zero forcing response"""
        identity = np.diag([1 for _ in range(0, self.__user_count)])
        freq_response = np.fft.fft(self.__response, axis=0)
        freq_length = len(freq_response)
        precoder = []
        for v in range(0, freq_length):
            factor = np.linalg.inv(freq_response[v] @ freq_response[v].conj().transpose() + self.__beta * identity)
            precoder.append(freq_response[v].transpose() @ factor)
        precoder = np.asarray(precoder)
        precoder = np.fft.ifft(precoder, axis=0)
        f = self.__normalization(precoder)
        precoder /= f
        precoder = np.asarray(precoder)
        return precoder

    def get_output(self, sym: np.ndarray) -> np.ndarray:
        """applying the regularized zero forcing to the symbols"""
        block_length = len(sym)
        precoder = self.get_response()
        vector = []
        for time in range(0, block_length):
            temp = []
            for tap in range(0, self.__length):
                index = (time + tap) % block_length
                temp.append(precoder[tap] @ sym[index])
            temp = np.asarray(temp)
            temp = np.sum(temp, axis=0)
            vector.append(temp)
        vector = np.asarray(vector)
        self.__desired_instance(precoder=precoder, sym=sym)
        return vector

    def __desired_instance(self, precoder: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """one instance of the desired symbols"""
        block_length = len(sym)
        temp = []
        for tap in range(0, self.__length):
            temp.append(self.__response[tap] @ precoder[tap])
        matrix = np.sum(temp, axis=0)
        instance = np.diag(matrix)
        self.__expected_instance.append(instance)
        desired = []
        for time in range(0, block_length):
            desired.append(matrix @ sym[time])
        desired = np.asarray(desired)
        self.__load_desired(desired=desired)
        return desired

    def __load_desired(self, desired: np.ndarray):
        """desired symbols over time"""
        self.__expected_matrix.append(desired)
        self.__count += 1
        return

    def get_expected(self) -> np.ndarray:
        """getting method for expected matrix"""
        return np.asarray(self.__expected_matrix)

    def get_instance(self) -> np.ndarray:
        """getting method for expected instance"""
        return np.asarray(self.__expected_instance)

    def get_gain(self) -> float:
        """desired signal power in theory"""
        signal = float(self.__antenna_count / self.__user_count)
        return signal

    def get_rate(self, factor: float, power: np.ndarray) -> np.ndarray:
        """achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * power
        noise = (factor / self.__antenna_count) * power + 1
        ratio = abs(signal / noise)
        rate = [(self.__user_count / 2) * np.log2(1 + element) for element in ratio]
        rate = np.asarray(rate)
        return rate

    def get_power(self, factor: float, rate: float) -> float:
        """required power in theory"""
        r = 2 * rate / self.__user_count
        temp = (self.__antenna_count * self.__user_count * (2 ** r - 1)) /\
               (self.__antenna_count * self.__antenna_count - self.__user_count * factor * (2 ** r - 1))
        power = 10 * np.log10(abs(temp))
        return power

    def get_limit(self, factor: float, max_power: float) -> float:
        """maximum achievable rate in theory"""
        signal = (self.__antenna_count / self.__user_count) * max_power
        noise = (factor / self.__antenna_count) * max_power + 1
        snr = abs(signal / noise)
        cap = (self.__user_count / 2) * np.log2(1 + snr)
        return cap
