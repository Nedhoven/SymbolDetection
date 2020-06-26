
import numpy as np

import src.main.Downlink as Dl
import src.main.Analyzer as An
import src.main.IO as IO

from pandas import DataFrame


points = 10
min_power = -1
max_power = 2

dn = 0.5

b_line = 0.0
t_line = (-0.36, 0.37)
q_line = 0.0
s_line = (-0.5, 0.0, 0.5)

opt = True


def set_power(total_points=points, power_min=min_power, power_max=max_power) -> bool:
    """setting global parameters"""
    global points, min_power, max_power
    try:
        points = total_points
        min_power = power_min
        max_power = power_max
        return True
    except TypeError:
        RuntimeError('set_power failed: unable to change parameters!')
        return False


def get_power() -> tuple:
    """getting power parameters"""
    return points, min_power, max_power


def get_lines() -> tuple:
    """getting boundary parameters"""
    return b_line, t_line, q_line, s_line


def set_distance(distance=dn) -> bool:
    """setting normalized distance"""
    global dn
    try:
        dn = distance
        return True
    except TypeError:
        RuntimeError('set_distance failed: unable to change parameters!')
        return False


def toggle_mode():
    """toggling operating mode"""
    global opt
    opt = not opt
    return


class DataGenerator:

    def __init__(self, antenna_count=128, user_count=10, length=4, factor=0, block_test=100000, block_train=100):
        """initializing data generator for downlink channel"""
        self.__antenna_count = antenna_count
        self.__user_count = user_count
        self.__length = length
        self.__factor = factor
        self.__block_test = block_test
        self.__block_train = block_train
        self.__model_train = Dl.DownlinkGenerator(antenna_count, user_count, length, block_train)
        self.__model_test = Dl.DownlinkGenerator(antenna_count, user_count, length, block_test)
        return

    def feature_test(self):
        """testing channels features"""
        data_points = self.__block_train * self.__user_count
        dl = Dl.DownlinkModule(self.__antenna_count, self.__user_count, self.__length, self.__block_train, None)
        decision_line = 0.0
        symbols = dl.get_symbol(mode='binary', line=decision_line)
        symbols = np.reshape(symbols, data_points)
        print()
        print('balanced symbols with line = ' + str(decision_line))
        print('--------------------------------------')
        print('symbols mean: ', end='')
        print(np.mean(symbols))
        print('symbols variance: ', end='')
        print(np.var(symbols))
        print('+1 symbol probability: ', end='')
        print(np.sum(symbols == +1) / data_points)
        print('0 symbol probability: ', end='')
        print(np.sum(symbols == 0) / data_points)
        print('--------------------------------------')
        decision_line = 0.4
        symbols = dl.get_symbol(mode='binary', line=decision_line)
        symbols = np.reshape(symbols, data_points)
        print()
        print('binary symbols with line = ' + str(decision_line))
        print('--------------------------------------')
        print('symbols mean: ', end='')
        print(np.mean(symbols))
        print('symbols variance: ', end='')
        print(np.var(symbols))
        print('0 symbol probability: ', end='')
        print(np.sum(symbols == 0) / data_points)
        print('1 symbol probability: ', end='')
        print(np.sum(symbols == 1) / data_points)
        decision_line = (-0.5, 0.5)
        symbols = dl.get_symbol(mode='ternary', line=decision_line)
        print('--------------------------------------')
        symbols = np.reshape(symbols, data_points)
        print()
        print('ternary symbols with line = ' + str(decision_line))
        print('--------------------------------------')
        print('symbols mean: ', end='')
        print(np.mean(symbols))
        print('symbols variance: ', end='')
        print(np.var(symbols))
        print('-1 symbol probability: ', end='')
        print(np.sum(symbols == -1) / data_points)
        print('0 symbol probability: ', end='')
        print(np.sum(symbols == 0) / data_points)
        print('+1 symbol probability: ', end='')
        print(np.sum(symbols == +1) / data_points)
        print('--------------------------------------')
        decision_line = 0.0
        symbols = dl.get_symbol(mode='quaternary', line=decision_line)
        print()
        print('quaternary symbols with line = ' + str(decision_line))
        print('--------------------------------------')
        print('symbols mean: ', end='')
        print(np.mean(symbols))
        print('symbols variance: ', end='')
        print(np.var(symbols))
        print('00 symbol probability: ', end='')
        print(np.sum((symbols[0] == 0) & (symbols[1] == 0)) / data_points)
        print('01 symbol probability: ', end='')
        print(np.sum((symbols[0] == 0) & (symbols[1] == 1)) / data_points)
        print('10 symbol probability: ', end='')
        print(np.sum((symbols[0] == 1) & (symbols[1] == 0)) / data_points)
        print('11 symbol probability: ', end='')
        print(np.sum((symbols[0] == 1) & (symbols[1] == 1)) / data_points)
        print('--------------------------------------')
        decision_line = (-0.5, 0.0, 0.5)
        symbols = dl.get_symbol(mode='sixteen', line=decision_line)
        print()
        print('sixteen symbols with line = ' + str(decision_line))
        print('--------------------------------------')
        print('symbols mean: ', end='')
        print(np.mean(symbols))
        print('symbols variance: ', end='')
        print(np.var(symbols))
        print('-3&-3 symbol probability: ', end='')
        print(np.sum((symbols[0] == -3) & (symbols[1] == -3)) / data_points)
        print('-3&-1 symbol probability: ', end='')
        print(np.sum((symbols[0] == -3) & (symbols[1] == -1)) / data_points)
        print('-3&+1 symbol probability: ', end='')
        print(np.sum((symbols[0] == -3) & (symbols[1] == +1)) / data_points)
        print('-3&+3 symbol probability: ', end='')
        print(np.sum((symbols[0] == -3) & (symbols[1] == +3)) / data_points)
        print('-1&-3 symbol probability: ', end='')
        print(np.sum((symbols[0] == -1) & (symbols[1] == -3)) / data_points)
        print('-1&-1 symbol probability: ', end='')
        print(np.sum((symbols[0] == -1) & (symbols[1] == -1)) / data_points)
        print('-1&+1 symbol probability: ', end='')
        print(np.sum((symbols[0] == -1) & (symbols[1] == +1)) / data_points)
        print('-1&+3 symbol probability: ', end='')
        print(np.sum((symbols[0] == -1) & (symbols[1] == +3)) / data_points)
        print('+1&-3 symbol probability: ', end='')
        print(np.sum((symbols[0] == +1) & (symbols[1] == -3)) / data_points)
        print('+1&-1 symbol probability: ', end='')
        print(np.sum((symbols[0] == +1) & (symbols[1] == -1)) / data_points)
        print('+1&+1 symbol probability: ', end='')
        print(np.sum((symbols[0] == +1) & (symbols[1] == +1)) / data_points)
        print('+1&+3 symbol probability: ', end='')
        print(np.sum((symbols[0] == +1) & (symbols[1] == +3)) / data_points)
        print('+3&-3 symbol probability: ', end='')
        print(np.sum((symbols[0] == +3) & (symbols[1] == -3)) / data_points)
        print('+3&-1 symbol probability: ', end='')
        print(np.sum((symbols[0] == +3) & (symbols[1] == -1)) / data_points)
        print('+3&+1 symbol probability: ', end='')
        print(np.sum((symbols[0] == +3) & (symbols[1] == +1)) / data_points)
        print('+3&+3 symbol probability: ', end='')
        print(np.sum((symbols[0] == +3) & (symbols[1] == +3)) / data_points)
        print('--------------------------------------')
        return

    def __get_comment_train(self, power: float) -> str:
        """generating comment for train data set"""
        text0 = 'Downlink'
        text1 = '\nMatched Filter'
        text2 = '\ncorrelation = NONE'
        text3 = '\nnumber of antennas = ' + str(self.__antenna_count)
        text4 = '\nnumber of users = ' + str(self.__user_count)
        text5 = '\nchannel length = ' + str(self.__length)
        text6 = '\ntransmission block length = ' + str(self.__block_train)
        text7 = '\n1 / (Noise Power) = ' + str(power)
        comment = text0 + text1 + text2 + text3 + text4 + text5 + text6 + text7
        return comment

    def __get_comment_test(self, power: float) -> str:
        """generating comment for test data set"""
        text0 = 'Downlink'
        text1 = '\nMatched Filter'
        text2 = '\ncorrelation = NONE'
        text3 = '\nnumber of antennas = ' + str(self.__antenna_count)
        text4 = '\nnumber of users = ' + str(self.__user_count)
        text5 = '\nchannel length = ' + str(self.__length)
        text6 = '\ntransmission block length = ' + str(self.__block_test)
        text7 = '\n1 / (Noise Power) = ' + str(power)
        comment = text0 + text1 + text2 + text3 + text4 + text5 + text6 + text7
        return comment

    def __create_train_data(self, power: float, mode: str):
        """creating data set to train models"""
        writer = IO.DataWriter()
        gen = self.__model_train
        if mode == 'binary':
            data_set = gen.record_data_set_binary(power=power, factors=self.__factor, line=b_line)
        elif mode == 'ternary':
            data_set = gen.record_data_set_ternary(power=power, factors=self.__factor, line=t_line)
        elif mode == 'quaternary':
            data_set = gen.record_data_set_quadrature(power=power, factors=self.__factor, line=q_line)
        else:
            data_set = gen.record_data_set_sixteen(power=power, factors=self.__factor, line=s_line)
        name = 'train_set'
        # comment = self.__get_comment_train(power=power)
        writer.frame_to_csv(data_set, name)
        # writer.comment_csv(comment, name)
        return

    def __create_test_data(self, powers, mode: str):
        """creating data set to test models"""
        writer = IO.DataWriter()
        gen = self.__model_test
        for power in powers:
            if mode == 'binary':
                data_set = gen.record_data_set_binary(power=power, factors=self.__factor, line=b_line)
            elif mode == 'ternary':
                data_set = gen.record_data_set_ternary(power=power, factors=self.__factor, line=t_line)
            elif mode == 'quaternary':
                data_set = gen.record_data_set_quadrature(power=power, factors=self.__factor, line=q_line)
            else:
                data_set = gen.record_data_set_sixteen(power=power, factors=self.__factor, line=s_line)
            name = 'test_' + ('%.2f' % power)
            # comment = self.__get_comment_test(power=power)
            writer.frame_to_csv(data_set, name)
            # writer.comment_csv(comment, name)
        return

    def run_channel_test(self, power: float):
        """recording channel runs"""
        gen = Dl.DownlinkGenerator(self.__antenna_count, self.__user_count, self.__length, self.__block_train)
        gen.data_run_test(power=power, factors=self.__factor)
        return

    def generate_train_data(self, mode: str, power=1.00):
        """running the channel model to obtain train data set"""
        self.__create_train_data(power=power, mode=mode)
        return

    def generate_test_data(self, mode: str):
        """running channel model to obtain test data set"""
        powers = np.logspace(min_power, max_power, points)
        self.__create_test_data(powers=powers, mode=mode)
        return


class TestLearner:

    def __init__(self, block_train=100):
        """initializing testing class"""
        self.__reader = IO.DataReader()
        self.__block_train = block_train
        return

    def __read_file(self, file: str, rows=None) -> DataFrame:
        """reading file"""
        data_set = self.__reader.read_data(data_name=file, rows=rows)
        return data_set

    def __get_data(self, file: str, labels=1, rows=None) -> tuple:
        """reading data set"""
        data_set = self.__read_file(file=file, rows=rows)
        if labels == 1:
            x_set = data_set.loc[:, data_set.columns == 'out']
            y_set = data_set.loc[:, data_set.columns == 'bit']
            y_set = y_set.values.flatten()
        else:
            x_set = data_set.loc[:, (data_set.columns != 'bit #1') & (data_set.columns != 'bit #2')]
            y_1 = data_set.loc[:, 'bit #1']
            y_2 = data_set.loc[:, 'bit #2']
            y_set = np.stack((y_1, y_2), axis=1)
        x_set = np.asarray(x_set)
        y_set = np.asarray(y_set)
        return x_set, y_set

    def __get_classifier(self, learner: str, file: str, bits: int, rows=None) -> An:
        """generating a linear learner"""
        if learner == 'ml':
            if bits == 2:
                analyzer = An.BinaryPredictor(line=b_line)
            elif bits == 3:
                x_set, y_set = self.__get_data(file=file, labels=1)
                gain = np.var(x_set) / np.var(y_set)
                analyzer = An.TernaryPredictor(line=t_line, gain=gain)
            elif bits == 4:
                analyzer = An.QuadraturePredictor(line=q_line)
            else:
                x_set, y_set = self.__get_data(file=file, labels=2)
                gain = np.mean(np.var(x_set, axis=0)) / np.mean(np.var(y_set, axis=0))
                analyzer = An.SixteenPredictor(line=s_line, gain=gain)
            return analyzer
        if learner == 'map':
            x_set, y_set = self.__get_data(file=file, labels=2)
            print('in __predictor_define -- ', end='')
            print('shape of y_set: ' + str(y_set.shape))
            total = y_set.shape[0]
            pos = float(np.sum(y_set > 0)) / total
            neg = float(np.sum(y_set < 0)) / total
            line = float(pos) / float(neg)
            analyzer = An.BinaryPredictor(line=line)
            return analyzer
        if learner == 'kmm':
            if bits == 2:
                analyzer = An.MeansClusterClassifier()
            elif bits == 4:
                analyzer = An.ParallelMeansClusterClassifier()
            else:
                raise Exception('learner=' + learner + ' does not accept ' + str(bits) + ' bits!')
            return analyzer
        if learner == 'acm':
            if bits == 2:
                analyzer = An.AgglomerativeClusterClassifier()
            elif bits == 4:
                analyzer = An.ParallelAgglomerativeClusterClassifier()
            else:
                raise Exception('learner=' + learner + ' does not accept ' + str(bits) + ' bits!')
            return analyzer
        if learner == 'gmm':
            if bits == 2:
                analyzer = An.GaussianMixtureClassifier()
            elif bits == 4:
                analyzer = An.ParallelGaussianMixtureClassifier()
            else:
                raise Exception('learner=' + learner + ' does not accept ' + str(bits) + ' bits!')
            return analyzer
        if bits == 4 or bits == 16:
            labels = 2
            if learner == 'bayes':
                analyzer = An.ParallelBayesClassifier()
            elif learner == 'logistic':
                analyzer = An.ParallelLogisticClassifier()
            elif learner == 'tree':
                analyzer = An.ParallelTreeClassifier()
            elif learner == 'forest':
                analyzer = An.ParallelForestClassifier()
            elif learner == 'vector':
                analyzer = An.ParallelVectorClassifier()
            elif learner == 'nearest':
                analyzer = An.ParallelNearestClassifier()
            else:
                raise Exception('learner=' + learner + ' is not defined!')
        else:
            labels = 1
            if learner == 'bayes':
                analyzer = An.BayesClassifier()
            elif learner == 'logistic':
                analyzer = An.LogisticClassifier()
            elif learner == 'tree':
                analyzer = An.TreeClassifier()
            elif learner == 'forest':
                analyzer = An.ForestClassifier()
            elif learner == 'vector':
                analyzer = An.VectorClassifier()
            elif learner == 'nearest':
                analyzer = An.NearestClassifier()
            else:
                raise Exception('learner=' + learner + ' is not defined!')
        x_set, y_set = self.__get_data(file=file, labels=labels, rows=rows)
        analyzer.fit(x_set, y_set)
        return analyzer

    def __test_classifier(self, analyzer: An, file: str, labels: int) -> tuple:
        """testing a blind predictor"""
        x_set, y_set = self.__get_data(file=file, labels=labels)
        score = analyzer.accuracy(x_set, y_set)
        return score, (1 - score)

    def get_error(self, learner: str, file: str, bits: int, labels: int, run: bool) -> np.ndarray:
        """getting error on learner"""
        if run:
            name = self.__get_classifier_identifier(learner=learner)
        else:
            name = None
        if learner == 'ml' or learner == 'map':
            return self.__error_blind(learner, file, bits, labels, name)
        if learner == 'gmm' or learner == 'kmm' or learner == 'acm':
            return self.__error_unsupervised(learner, file, bits, labels, name)
        return self.__error_supervised(learner, file, bits, labels, name)

    def __error_blind(self, learner: str, file: str, bits: int, labels: int, name: str) -> np.ndarray:
        """getting error on blind learner"""
        powers = np.logspace(min_power, max_power, points)
        error = np.ones(len(powers))
        for index in range(0, len(powers)):
            power = powers[index]
            test = file + 'test_' + ('%.2f' % power)
            model = self.__get_classifier(learner=learner, file=test, bits=bits)
            err = self.__test_classifier(analyzer=model, file=test, labels=labels)
            if name is not None:
                self.__print_info(name=name, power=power, error=err[1])
            error[index] = err[1]
        return error

    def __error_supervised(self, learner: str, file: str, bits: int, labels: int, name: str) -> np.ndarray:
        """getting error on supervised learner"""
        if opt:
            return self.__error__supervised(learner, file, bits, labels, name)
        powers = np.logspace(min_power, max_power, points)
        train = file + 'test_100.00'
        model = self.__get_classifier(learner=learner, file=train, bits=bits, rows=self.__block_train)
        error = np.ones(len(powers))
        for index in range(0, len(powers)):
            power = powers[index]
            test = file + 'test_' + ('%.2f' % power)
            err = self.__test_classifier(analyzer=model, file=test, labels=labels)
            if name is not None:
                self.__print_info(name=name, power=power, error=err[1])
            error[index] = err[1]
        return error

    def __error__supervised(self, learner: str, file: str, bits: int, labels: int, name: str) -> np.ndarray:
        """getting optimum error on supervised learner"""
        powers = np.logspace(min_power, max_power, points)
        error = np.ones(len(powers))
        for index in range(0, len(powers)):
            power = powers[index]
            test = file + 'test_' + ('%.2f' % power)
            model = self.__get_classifier(learner=learner, file=test, bits=bits, rows=self.__block_train)
            err = self.__test_classifier(analyzer=model, file=test, labels=labels)
            if name is not None:
                self.__print_info(name=name, power=power, error=err[1])
            error[index] = err[1]
        return error

    def __error_unsupervised(self, learner: str, file: str, bits: int, labels: int, name: str) -> np.ndarray:
        """getting error on unsupervised learner"""
        powers = np.logspace(min_power, max_power, points)
        error = np.ones(len(powers))
        for index in range(0, len(powers)):
            power = powers[index]
            test = file + 'test_' + ('%.2f' % power)
            model = self.__get_classifier(learner=learner, file=test, bits=bits, rows=self.__block_train)
            err = self.__test_classifier(analyzer=model, file=test, labels=labels)
            if name is not None:
                self.__print_info(name=name, power=power, error=err[1])
            error[index] = err[1]
        return error

    @staticmethod
    def __get_classifier_identifier(learner: str) -> str:
        """getting classifier's identifier"""
        if learner == 'ml':
            name = 'Maximum Likelihood Predictor'
        elif learner == 'map':
            name = 'Maximum A Posterior Predictor'
        elif learner == 'bayes':
            name = 'Naive Bayes Classifier'
        elif learner == 'logistic':
            name = 'Logistic Regression Classifier'
        elif learner == 'tree':
            name = 'Decision Tree Classifier'
        elif learner == 'forest':
            name = 'Random Forests Classifier'
        elif learner == 'vector':
            name = 'Support Vector Machines Classifier'
        elif learner == 'nearest':
            name = 'Nearest Neighbors Classifier'
        elif learner == 'gmm':
            name = 'Gaussian Mixtures Classifier'
        elif learner == 'acm':
            name = 'Agglomerative Clustering Classifier'
        else:
            name = 'Means Clustering Classifier'
        return name

    @staticmethod
    def __print_info(name: str, power: float, error: float):
        """printing run down of results"""
        print('with 1 / En = ' + ('%.2f' % power), end=': ')
        print(name + ' Error = ' + str(error))
        return
