
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering as Ac


def get_symbols(user=10, block=10000):
    sym = np.random.uniform(-1, 1, size=(user * block))
    sym = np.where(sym >= 0, 1, sym)
    sym = np.where(sym < 0, -1, sym)
    sym = np.asarray(np.reshape(1 * sym, newshape=(user * block)))
    return sym


def reshape(data):
    data = np.reshape(data, newshape=(data.shape[0], 1))
    return data


def get_centers(labels: np.ndarray, x_set: np.ndarray):
    res = np.zeros(shape=2)
    for index in range(0, 2):
        temp = np.where(labels == index)
        mean = np.mean(x_set[temp])
        res[index] = mean
    if res[0] < res[1]:
        res[0] = -1
        res[1] = +1
    else:
        res[0] = +1
        res[1] = -1
    return res


def predict(centers, y_hat):
    if centers[0] > centers[1]:
        y_hat = np.where(y_hat == 1, centers[1], y_hat)
        y_hat = np.where(y_hat == 0, centers[0], y_hat)
    else:
        y_hat = np.where(y_hat == 0, centers[0], y_hat)
        y_hat = np.where(y_hat == 1, centers[1], y_hat)
    return y_hat


def get_error(y_hat, y_set):
    score = np.sum(y_hat == y_set)
    score = score / y_set.shape[0]
    return (1 - score), score


def run():
    y = get_symbols(user=10, block=5000)
    x = reshape(y)
    model = Ac(n_clusters=2)
    y_hat = model.fit_predict(x)
    centers = get_centers(labels=model.labels_, x_set=x)
    y_hat = predict(centers=centers, y_hat=y_hat)
    err = get_error(y_hat=y_hat, y_set=y)
    return err


def test_run():
    start_time = time.time()
    result = run()
    stop_time = time.time()
    print(stop_time - start_time)
    print('error, accuracy: ', end='')
    print(result)
    return
