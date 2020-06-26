
import time
import numpy as np

import src.main.IO as IO
import src.main.Manager as Mg


def read_file(file: str, rows=None):
    """reading file"""
    reader = IO.DataReader()
    data_set = reader.read_data(data_name=file, rows=rows)
    return data_set


def downlink_test_run(flag: bool):
    """running the channel to test features"""
    antennas = 128
    users = 10
    length = 4
    power = 1.0
    if flag:
        manager = Mg.DataGenerator(antenna_count=antennas, user_count=users, length=length)
        manager.feature_test()
    else:
        manager = Mg.DataGenerator(antenna_count=antennas, user_count=users, length=length)
        manager.run_channel_test(power=power)
    return


def get_distribution(mode='quaternary', path='try_1/', file='test_1.00', rows=300, alpha=1.0, labeled=False) -> bool:
    """visualizing the distribution of data"""
    if mode == 'sixteen':
        return __get_distribution_sixteen(path=path, file=file, rows=rows, alpha=alpha, labeled=labeled)
    if mode != 'quaternary':
        RuntimeError('mode=' + mode + ' is not an option!')
        return False
    data = mode + '/' + path + file
    data_set = read_file(file=data, rows=rows)
    x1 = data_set['out #1']
    x2 = data_set['out #2']
    plot = IO.Plotter(dark=False, x_title='bit #1', y_title='bit #2')
    if labeled:
        y1 = data_set['bit #1']
        y2 = data_set['bit #2']
        s0 = -1
        s1 = +1
        index_0 = np.where((y1 == s0) & (y2 == s0))[0]
        index_1 = np.where((y1 == s0) & (y2 == s1))[0]
        index_2 = np.where((y1 == s1) & (y2 == s0))[0]
        index_3 = np.where((y1 == s1) & (y2 == s1))[0]
        plot.to_scatter(x1[index_0], x2[index_0], edge_color='purple', label='00', marker='o', alpha=alpha)
        plot.to_scatter(x1[index_1], x2[index_1], edge_color='red', label='01', marker='^', alpha=alpha)
        plot.to_scatter(x1[index_2], x2[index_2], edge_color='blue', label='10', marker='d', alpha=alpha)
        plot.to_scatter(x1[index_3], x2[index_3], edge_color='green', label='11', marker='s', alpha=alpha)
    else:
        plot.to_scatter(x1, x2, edge_color='black', marker='o', alpha=alpha, in_color='blue')
    plot.show()
    return True


def __get_distribution_sixteen(path: str, file: str, rows: int, alpha: float, labeled: bool) -> bool:
    """visualizing the distribution of data"""
    data = 'sixteen/' + path + file
    data_set = read_file(file=data, rows=rows)
    x1 = data_set['out #1']
    x2 = data_set['out #2']
    plot = IO.Plotter(dark=False, x_title='bit #1', y_title='bit #2')
    if labeled:
        y1 = data_set['bit #1']
        y2 = data_set['bit #2']
        s00 = -3
        s01 = -1
        s10 = +1
        s11 = +3
        index_00 = np.where((y1 == s00) & (y2 == s00))[0]
        index_01 = np.where((y1 == s00) & (y2 == s01))[0]
        index_02 = np.where((y1 == s00) & (y2 == s10))[0]
        index_03 = np.where((y1 == s00) & (y2 == s11))[0]
        index_04 = np.where((y1 == s01) & (y2 == s00))[0]
        index_05 = np.where((y1 == s01) & (y2 == s01))[0]
        index_06 = np.where((y1 == s01) & (y2 == s10))[0]
        index_07 = np.where((y1 == s01) & (y2 == s11))[0]
        index_08 = np.where((y1 == s10) & (y2 == s00))[0]
        index_09 = np.where((y1 == s10) & (y2 == s01))[0]
        index_10 = np.where((y1 == s10) & (y2 == s10))[0]
        index_11 = np.where((y1 == s10) & (y2 == s11))[0]
        index_12 = np.where((y1 == s11) & (y2 == s00))[0]
        index_13 = np.where((y1 == s11) & (y2 == s01))[0]
        index_14 = np.where((y1 == s11) & (y2 == s10))[0]
        index_15 = np.where((y1 == s11) & (y2 == s11))[0]
        c = np.random.uniform(0, 1, size=(16, 3))
        plot.to_scatter(x1[index_00], x2[index_00], edge_color=c[0], label='0000', marker='v', alpha=alpha)
        plot.to_scatter(x1[index_01], x2[index_01], edge_color=c[1], label='0001', marker='o', alpha=alpha)
        plot.to_scatter(x1[index_02], x2[index_02], edge_color=c[2], label='0010', marker='*', alpha=alpha)
        plot.to_scatter(x1[index_03], x2[index_03], edge_color=c[3], label='0011', marker='p', alpha=alpha)
        plot.to_scatter(x1[index_04], x2[index_04], edge_color=c[4], label='0100', marker='s', alpha=alpha)
        plot.to_scatter(x1[index_05], x2[index_05], edge_color=c[5], label='0101', marker='<', alpha=alpha)
        plot.to_scatter(x1[index_06], x2[index_06], edge_color=c[6], label='0110', marker='P', alpha=alpha)
        plot.to_scatter(x1[index_07], x2[index_07], edge_color=c[7], label='0111', marker='h', alpha=alpha)
        plot.to_scatter(x1[index_08], x2[index_08], edge_color=c[8], label='1000', marker='o', alpha=alpha)
        plot.to_scatter(x1[index_09], x2[index_09], edge_color=c[9], label='1001', marker='D', alpha=alpha)
        plot.to_scatter(x1[index_10], x2[index_10], edge_color=c[10], label='1010', marker='^', alpha=alpha)
        plot.to_scatter(x1[index_11], x2[index_11], edge_color=c[11], label='1011', marker='8', alpha=alpha)
        plot.to_scatter(x1[index_12], x2[index_12], edge_color=c[12], label='1100', marker='X', alpha=alpha)
        plot.to_scatter(x1[index_13], x2[index_13], edge_color=c[13], label='1101', marker='d', alpha=alpha)
        plot.to_scatter(x1[index_14], x2[index_14], edge_color=c[14], label='1110', marker='*', alpha=alpha)
        plot.to_scatter(x1[index_15], x2[index_15], edge_color=c[15], label='1111', marker='>', alpha=alpha)
    else:
        plot.to_scatter(x1, x2, edge_color='black', marker='o', in_color='blue', alpha=alpha)
    plot.show()
    return True


def create_train_set(antenna_count=128, user_count=10, length=4, mode='quaternary', power=1.00):
    """creating train data set using Manager interface"""
    manager = Mg.DataGenerator(antenna_count=antenna_count, user_count=user_count, length=length)
    manager.generate_train_data(mode=mode, power=power)
    return


def create_test_set(antenna_count=128, user_count=10, length=4, mode='quaternary'):
    """creating test data set using Manager interface"""
    manager = Mg.DataGenerator(antenna_count=antenna_count, user_count=user_count, length=length, block_test=100000)
    manager.generate_test_data(mode=mode)
    return


def visualize_error(learner: str, mode: str, path='try_1/', run=False, block=100) -> bool:
    """visualizing error"""
    manager = Mg.TestLearner(block_train=block)
    if mode == 'binary':
        labels = 1
        bits = 2
    elif mode == 'ternary':
        labels = 1
        bits = 3
    elif mode == 'quaternary':
        labels = 2
        bits = 4
    elif mode == 'sixteen':
        labels = 2
        bits = 16
    else:
        RuntimeError('mode=' + mode + ' is not defined!')
        return False
    file = mode + '/' + path
    data = manager.get_error(learner=learner, file=file, bits=bits, labels=labels, run=run)
    if learner == 'ml':
        title = 'Maximum Likelihood'
    elif learner == 'map':
        title = 'Maximum A Posterior'
    elif learner == 'bayes':
        title = 'Bayes Classifier'
    elif learner == 'logistic':
        title = 'Logistic Regression'
    elif learner == 'tree':
        title = 'Decision Tree Classifier'
    elif learner == 'vector':
        title = 'Support Vector Machines Classifier'
    elif learner == 'forest':
        title = 'Random Forests Classifier'
    elif learner == 'nnet':
        title = 'Neural Networks Classifier'
    else:
        RuntimeError('learner=' + learner + ' is not defined')
        return False
    points, min_power, max_power = Mg.get_power()
    plot = IO.Plotter(dark=True, x_title='1 / En (dB)', y_title='Error')
    power_set = np.logspace(min_power, max_power, points)
    power_set = [10 * np.log10(x_power) for x_power in power_set]
    plot.to_log(x=power_set, y=data, color='blue', label=title)
    plot.show()
    return True


def get_performance(learner: str, mode: str, path: str, block=100) -> np.ndarray:
    """obtaining detection performance"""
    manager = Mg.TestLearner(block_train=block)
    if mode == 'binary':
        labels = 1
        bits = 2
    elif mode == 'ternary':
        labels = 1
        bits = 3
    elif mode == 'quaternary':
        labels = 2
        bits = 4
    elif mode == 'sixteen':
        labels = 2
        bits = 16
    else:
        raise Exception('mode=' + mode + ' is invalid!')
    file = mode + '/' + path
    error = manager.get_error(learner=learner, file=file, bits=bits, labels=labels, run=False)
    return error


def get_specifications(learner: str, mode: str) -> tuple:
    """obtaining runtime"""
    res_time = []
    res_error = []
    block_length_list = np.logspace(1, 6, 6, dtype='int')
    paths_list = np.array(('try_1/', 'try_2/', 'try_3/', 'try_4/', 'try_5/'))
    for path_local in paths_list:
        time_local = []
        error_local = []
        print('processing path "' + path_local + '":')
        for length_local in block_length_list:
            print('\tprocessing length: ' + str(length_local))
            start_time = time.time()
            err_local = get_performance(learner=learner, mode=mode, path=path_local, block=length_local)
            stop_time = time.time()
            time_stamp = stop_time - start_time
            time_local.append(time_stamp)
            error_local.append(err_local)
        time_local = np.asarray(time_local)
        res_time.append(time_local)
        error_local = np.asarray(error_local)
        res_error.append(error_local)
    res_time = np.asarray(res_time)
    res_error = np.asarray(res_error)
    return res_error, res_time


if __name__ == '__main__':
    """main derive"""
    # create_test_set(antenna_count=128, user_count=10, length=4, mode='quaternary')
    # visualize_error(learner='vector', mode='quaternary', path='try_2/', run=True)
    # perform = get_performance(learner='vector', mode='quaternary', path='try_2/')
    # get_distribution(mode='quaternary', path='try_1/', file='test_0.10', rows=500, alpha=1.0)
    # total_result = get_specifications(learner='gmm', mode='quaternary')
    # print('error:')
    # print(total_result[0])
    # print()
    # print('time:')
    # print(total_result[1])

