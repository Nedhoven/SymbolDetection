
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt


def dft_test():
    h = np.random.normal(0, 1, (4, 8, 6))
    print(h.shape)
    w = np.fft.fft(h, axis=0, n=20)
    print(w.shape)
    h_prime = np.fft.ifft(w, axis=0, n=4)
    print(h_prime.shape)
    count = 0
    for i in range(0, 4):
        for j in range(0, 8):
            for k in range(0, 6):
                if h[i][j][k] != h_prime[i][j][k]:
                    print('error in index (' + str(i) + ', ' + str(j) + ', ' + str(k) + ')!')
                    print('obtained: ' + str(h_prime[i][j][k]) + ', but was: ' + str(h[i][j][k]))
                    count += 1
    print('error count: ' + str(count) + ' / ' + str(4 * 8 * 6))
    return


def matrix_test():
    matrix1 = np.array([[1, 2], [4, 5]])
    matrix2 = np.array([[1, 2], [3, 4], [5, 6]])
    print(matrix1.shape)
    print(matrix2.shape)
    nader = np.linalg.inv(matrix1)
    print(matrix1 @ nader)
    return


def data_test():
    matrix = np.random.normal(0, 1, (2, 3, 4))
    print(matrix)
    mt = matrix.reshape((6, 4))
    print(mt)
    return


def diagonal_test():
    temp = [1 for _ in range(0, 5)]
    matrix = np.diag(temp)
    print(matrix)
    return


def array_test():
    temp = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    print(len(temp))
    print(temp[0:10])
    return


def check_plot():
    x_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig = plt
    fig.plot(x_test, y_test)
    fig.show()
    return


def array_pos():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    b = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    c = np.sum(a == b)
    print(c)
    return


def data_set_test():
    x_test_1 = np.random.uniform(0, 10, size=10)
    x_test_2 = np.random.uniform(0, 10, size=10)
    x_test_3 = np.random.uniform(0, 10, size=10)
    y_test_1 = np.random.randint(0, 2, size=10)
    y_test_2 = np.random.randint(0, 2, size=10)
    data = {'bit 1': x_test_1, 'bit 2': x_test_2, 'bit 3': x_test_3, 'label 1': y_test_1, 'label 2': y_test_2}
    df = DataFrame(data)
    labels = df[['label 1', 'label 2']]
    df.drop(columns=['label 1', 'label 2'], inplace=True)
    print(labels)
    print(df)
    scalar = preprocessing.StandardScaler()
    scalar.fit(df)
    df = DataFrame(scalar.transform(df))
    print(df)
    return


def random_test():
    x_test = np.random.uniform(0, 1, size=(1000, 100))
    x_test = np.where(x_test >= 0.69, 2, x_test)
    x_test = np.where(x_test < 0.62, -1, x_test)
    x_test = np.where((x_test >= 0.62) & (x_test < 0.69), 0, x_test)
    x_test = np.reshape(x_test, (1000, 100))
    print(np.sum(x_test == 0) / 100000)
    print(np.sum(x_test == -1) / 100000)
    print(np.sum(x_test == 2) / 100000)
    return
