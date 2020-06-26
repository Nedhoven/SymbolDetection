
import numpy as np

from pandas import DataFrame

import src.main.Structure as Str
import src.main.Precoder as Pr


dn = 0.5


def set_distance(dist=dn):
    """change normalized distance"""
    global dn
    dn = dist
    return


class SymbolGenerator:

    def __init__(self, num_rows: int, num_cols: int, s0=-1, s1=1, sm=0, s00=-3, s01=-1, s10=1, s11=3, signal_var=1.00):
        """initializing symbol generator"""
        self.__row_size = num_rows
        self.__col_size = num_cols
        self.__s0 = s0
        self.__s1 = s1
        self.__sm = sm
        self.__s00 = s00
        self.__s01 = s01
        self.__s10 = s10
        self.__s11 = s11
        self.__std = np.sqrt(signal_var)
        return

    def unary(self) -> np.ndarray:
        """unary random symbols"""
        std = np.sqrt(1 / 2)
        rl = np.random.normal(0, std, size=(self.__row_size, self.__col_size))
        im = np.random.normal(0, std, size=(self.__row_size, self.__col_size))
        sym = 1j * im + rl
        sym = np.asarray(self.__std * sym)
        return sym

    def binary(self, line=0.0) -> np.ndarray:
        """binary random symbols"""
        sym = np.random.uniform(-1, 1, size=(self.__row_size, self.__col_size))
        sym = np.where(sym < line, self.__s0, sym)
        sym = np.where(sym >= line, self.__s1, sym)
        sym = np.reshape(self.__std * sym, (self.__row_size, self.__col_size))
        return sym

    def ternary(self, line=(-0.36, 0.37)) -> np.ndarray:
        """ternary random symbols"""
        sym = np.random.uniform(-1, 1, size=(self.__row_size, self.__col_size))
        sym = np.where(sym >= line[1], self.__s1, sym)
        sym = np.where(sym < line[0], self.__s0, sym)
        sym = np.where((sym >= line[0]) & (sym < line[1]), self.__sm, sym)
        sym = np.reshape(self.__std * sym, (self.__row_size, self.__col_size))
        return sym

    def quaternary(self, line=0.0) -> np.ndarray:
        """quaternary random symbols"""
        sym = np.random.uniform(-1, 1, size=(2, self.__row_size, self.__col_size))
        sym = np.where(sym >= line, self.__s1, sym)
        sym = np.where(sym < line, self.__s0, sym)
        sym = np.reshape(self.__std * sym, (2, self.__row_size, self.__col_size))
        return sym

    def sixteen(self, line=(-0.5, 0, 0.5)) -> np.ndarray:
        """array of random symbols"""
        sym = np.random.uniform(-1, 1, size=(2, self.__row_size, self.__col_size))
        sym = np.where(sym < line[0], self.__s00, sym)
        sym = np.where(sym >= line[2], self.__s11, sym)
        sym = np.where((sym < line[1]) & (sym >= line[0]), self.__s01, sym)
        sym = np.where((sym < line[2]) & (sym >= line[1]), self.__s10, sym)
        sym = np.reshape(self.__std * sym, (2, self.__row_size, self.__col_size))
        return sym


class ChannelModule:

    def __init__(self, antenna_count: int, user_count: int, length: int):
        """initializing the downlink channel"""
        self.__response = []
        self.__antenna_count = antenna_count
        self.__user_count = user_count
        self.__length = length
        return

    def __profile_generator(self) -> np.ndarray:
        """power delay profile generator"""
        theta = np.array([p_temp / 5 for p_temp in range(0, self.__user_count)])
        depth_local = np.array([d_temp for d_temp in range(0, self.__length)])
        power_profile = []
        for depth in range(0, self.__length):
            power_local = np.zeros(shape=self.__user_count)
            for user in range(0, self.__user_count):
                delay_temp = np.exp(-1 * theta[user] * depth_local)
                power_local[user] = np.exp(-1 * theta[user] * depth) / np.sum(delay_temp)
            profile_temp = np.diag(power_local)
            power_profile.append(profile_temp)
        power_profile = np.array(power_profile)
        return power_profile

    def get_response(self, corr: np.ndarray, res_var=1.00) -> np.ndarray:
        """response matrix generator"""
        self.__response = []
        pdp = self.__profile_generator()
        std = np.sqrt(res_var / 2)
        for tap in range(0, self.__length):
            rl = np.random.normal(0, std, size=(self.__antenna_count, self.__user_count))
            im = np.random.normal(0, std, size=(self.__antenna_count, self.__user_count))
            res = np.array(1j * im + rl)
            temp = (np.power(corr, 0.5) @ res @ np.power(pdp[tap], 0.5)).transpose().conj()
            self.__response.append(temp)
        response = np.array(self.__response)
        return response

    def __get_noise(self, block_length: int, noise_var: float):
        """generating white additive Gaussian noise"""
        std = np.sqrt(noise_var / 2)
        rl = np.random.normal(0, std, size=(self.__length, block_length, self.__user_count))
        im = np.random.normal(0, std, size=(self.__length, block_length, self.__user_count))
        noise = np.array(1j * im + rl)
        return noise

    def get_output(self, vec: np.ndarray, noise_var: float) -> np.ndarray:
        """applying channel on the input symbols"""
        block_length = len(vec)
        noise = self.__get_noise(block_length=block_length, noise_var=noise_var)
        res = np.array(self.__response)
        received = []
        for time in range(0, block_length):
            temp = []
            for tap in range(0, self.__length):
                index = time - tap
                temp.append(res[tap] @ vec[index] + noise[tap][time])
            received.append(np.sum(temp, axis=0))
        received = np.asarray(received)
        return received


class DownlinkModule:

    def __init__(self, antenna_count: int, user_count: int, length: int, block: int, precoder: Pr):
        """initializing the frequency selective downlink generator"""
        self.__antenna_count = antenna_count
        self.__user_count = user_count
        self.__channel_length = length
        self.__block_length = block
        self.__structure = Str.Array(dn=dn)
        self.__correlation = Str.Correlation()
        self.__channel = ChannelModule(antenna_count=antenna_count, user_count=user_count, length=length)
        self.__precoder = precoder
        self.__rand = SymbolGenerator(num_rows=block, num_cols=user_count)
        return

    def get_symbol(self, mode: str, line) -> np.ndarray:
        """symbol generator"""
        if mode == 'unary':
            sym = self.__rand.unary()
        elif mode == 'binary':
            sym = self.__rand.binary(line=line)
        elif mode == 'ternary':
            sym = self.__rand.ternary(line=line)
        elif mode == 'quaternary':
            sym = self.__rand.quaternary(line=line)
        else:
            sym = self.__rand.sixteen(line=line)
        return sym

    def get_response(self, factors) -> np.ndarray:
        """modeling the channel response"""
        dist = self.__structure.linear(self.__antenna_count)
        if type(factors) == int or type(factors) == float:
            self.__correlation = Str.Correlation(alpha=factors)
            corr = self.__correlation.exponential(dist)
        elif type(factors) == list and len(factors) == 2:
            eta = factors[0]
            mu = factors[1]
            self.__correlation = Str.Correlation(eta=eta, mu=mu)
            corr = self.__correlation.bessel(dist)
        else:
            raise Exception('wrong factors in get_channel!')
        res = self.__channel.get_response(corr=corr)
        return res

    def get_input(self, res: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """input of downlink channel"""
        in_vec = self.__precoder.get_output(res=res, sym=sym)
        return in_vec

    def get_desired(self) -> np.ndarray:
        """returning the desired symbols"""
        desired = self.__precoder.get_expected()
        return desired

    def get_desired_instance(self) -> np.ndarray:
        """returning one instances of the desired symbols"""
        desired = self.__precoder.get_instance()
        return desired

    def get_output(self, vec: np.ndarray, noise_var: float) -> np.ndarray:
        """result of the downlink channel"""
        received = self.__channel.get_output(vec=vec, noise_var=noise_var)
        return received


class DownlinkGenerator:

    def __init__(self, antenna_count: int, user_count: int, channel_length: int, block_length: int):
        """initializing the data generator class"""
        self.__antenna_count = antenna_count
        self.__user_count = user_count
        self.__channel_length = channel_length
        self.__block_length = block_length
        precoder = Pr.MatchedFilter(antenna_count, user_count, channel_length)
        self.__channel = DownlinkModule(antenna_count, user_count, channel_length, block_length, precoder)
        return

    def get_rate(self, iteration: int, power: float, factors) -> float:
        """single bit rate for a given input power"""
        noise_var = 1 / power
        out_matched = []
        for _ in range(0, iteration):
            response = self.__channel.get_response(factors=factors)
            symbols = self.__channel.get_symbol(mode='unary', line=0)
            in_matched = self.__channel.get_input(res=response, sym=symbols)
            output = self.__channel.get_output(vec=in_matched, noise_var=noise_var)
            out_matched.append(output)
        effective_noise = []
        desired = self.__channel.get_desired()
        desired = np.asarray(desired)
        desired = np.mean(desired, axis=0)
        for i in range(0, iteration):
            temp = out_matched[i] - desired
            effective_noise.append(temp)
        noise = np.var(np.mean(effective_noise, axis=0), axis=0)
        ratio = np.var(desired, axis=0) / noise
        bit = [0.5 * np.log2(1 + x) for x in ratio]
        rate = float(np.sum(bit))
        return rate

    def general_channel_data(self, iteration: int, power: float, factors) -> DataFrame:
        """channel information recorder"""
        noise_var = 1 / power
        data_set = DataFrame()
        for index in range(0, iteration):
            response = self.__channel.get_response(factors=factors)
            symbols = self.__channel.get_symbol(mode='binary', line=0.5)
            feed = self.__channel.get_input(res=response, sym=symbols)
            output = self.__channel.get_output(vec=feed, noise_var=noise_var)
            if index == 0:
                data_set = self.__data_flatten(res=response, sym=symbols, vec=feed, out=output)
            else:
                data_set.append(self.__data_flatten(response, symbols, feed, output))
        return data_set

    def record_data_set_binary(self, power: float, factors, line: float) -> DataFrame:
        """channel run to create a data set for training"""
        noise_var = 1 / power
        symbols = self.__channel.get_symbol(mode='binary', line=line)
        y_set = self.__data_reform(sym=symbols)
        response = self.__channel.get_response(factors=factors)
        feed = self.__channel.get_input(res=response, sym=symbols)
        output = self.__channel.get_output(vec=feed, noise_var=noise_var)
        x_set = self.__data_reform(sym=output)
        data = {'out': x_set, 'bit': y_set}
        df = DataFrame(data)
        return df

    def record_data_set_ternary(self, power: float, factors, line: tuple) -> DataFrame:
        """channel run to create a data set for training"""
        noise_var = 1 / power
        symbols = self.__channel.get_symbol(mode='ternary', line=line)
        y_set = self.__data_reform(sym=symbols)
        response = self.__channel.get_response(factors=factors)
        feed = self.__channel.get_input(res=response, sym=symbols)
        output = self.__channel.get_output(vec=feed, noise_var=noise_var)
        x_set = self.__data_reform(sym=output)
        data = {'out': x_set, 'bit': y_set}
        df = DataFrame(data)
        return df

    def record_data_set_quadrature(self, power: float, factors, line: float) -> DataFrame:
        """channel run to create a data set for training"""
        noise_var = 1 / power
        symbols = self.__channel.get_symbol(mode='quaternary', line=line)
        y_set_1 = self.__data_reform(sym=symbols[0])
        y_set_2 = self.__data_reform(sym=symbols[1])
        response = self.__channel.get_response(factors=factors)
        feed_1 = self.__channel.get_input(res=response, sym=symbols[0])
        feed_2 = self.__channel.get_input(res=response, sym=symbols[1])
        output_1 = self.__channel.get_output(vec=feed_1, noise_var=noise_var)
        output_2 = self.__channel.get_output(vec=feed_2, noise_var=noise_var)
        x_set_1 = self.__data_reform(sym=output_1)
        x_set_2 = self.__data_reform(sym=output_2)
        data = {'out #1': x_set_1, 'out #2': x_set_2, 'bit #1': y_set_1, 'bit #2': y_set_2}
        df = DataFrame(data)
        return df

    def record_data_set_sixteen(self, power: float, factors, line: tuple) -> DataFrame:
        """channel run to create a data set for training"""
        noise_var = 1 / power
        symbols = self.__channel.get_symbol(mode='sixteen', line=line)
        y_set_1 = self.__data_reform(sym=symbols[0])
        y_set_2 = self.__data_reform(sym=symbols[1])
        response = self.__channel.get_response(factors=factors)
        feed_1 = self.__channel.get_input(res=response, sym=symbols[0])
        feed_2 = self.__channel.get_input(res=response, sym=symbols[1])
        output_1 = self.__channel.get_output(vec=feed_1, noise_var=noise_var)
        output_2 = self.__channel.get_output(vec=feed_2, noise_var=noise_var)
        x_set_1 = self.__data_reform(sym=output_1)
        x_set_2 = self.__data_reform(sym=output_2)
        data = {'out #1': x_set_1, 'out #2': x_set_2, 'bit #1': y_set_1, 'bit #2': y_set_2}
        df = DataFrame(data)
        return df

    def data_run_test(self, power: float, factors) -> bool:
        """channel run to create a data set for testing"""
        noise_var = 1 / power
        symbols = self.__channel.get_symbol(mode='binary', line=0.0)
        print('symbols shape: ', end='')
        print(symbols.shape)
        print('symbols: ', end='')
        print(symbols)
        y_set = self.__data_reform(symbols)
        print('labels shape: ', end='')
        print(y_set.shape)
        response = self.__channel.get_response(factors)
        print('response shape: ', end='')
        print(response.shape)
        feed = self.__channel.get_input(response, symbols)
        print('input shape: ', end='')
        print(feed.shape)
        output = self.__channel.get_output(vec=feed, noise_var=noise_var)
        print('output shape: ', end='')
        print(output.shape)
        x_set = self.__data_reform(output)
        print('training sequence shape: ', end='')
        print(x_set.shape)
        return True

    def __data_flatten(self, res: np.ndarray, sym: np.ndarray, vec: np.ndarray, out: np.ndarray) -> DataFrame:
        """creating a flat data structure"""
        user_list = []
        channel_list = []
        tap_list = []
        antenna_list = []
        time_list = []
        in_list = []
        out_list = []
        sent_list = []
        for user in range(0, self.__user_count):
            for tap in range(0, self.__channel_length):
                for ant in range(0, self.__antenna_count):
                    for time in range(0, self.__block_length):
                        user_list.append(user)
                        tap_list.append(tap)
                        antenna_list.append(ant)
                        sent_list.append(vec[time][ant])
                        channel_list.append(res[tap][user][ant])
                        time_list.append(time)
                        in_list.append(sym[time][user])
                        out_list.append(out[time][user])
        result = {'user': user_list, 'tap': tap_list, 'antenna': antenna_list, 'channel': channel_list,
                  'time': time_list, 'sent': sent_list, 'input': in_list, 'output': out_list}
        df = DataFrame(result, columns=['user', 'tap', 'antenna', 'channel', 'time', 'sent', 'input', 'output'])
        return df

    def __data_reform(self, sym: np.ndarray) -> np.ndarray:
        """flatten one single entry"""
        data = np.reshape(arrays=sym, newshape=self.__user_count * self.__block_length)
        data = np.sign(data).real * np.abs(data)
        data = np.asarray(data)
        return data
