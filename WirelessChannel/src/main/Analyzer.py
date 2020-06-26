
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix


class SupervisedClassifier(object):

    def __init__(self, model):
        """initializing supervised classifier"""
        self.__model = model
        return

    def fit(self, x_set: np.ndarray, y_set: np.ndarray):
        """training model"""
        x_set = self.__scale(data=x_set)
        self.__model.fit(x_set, y_set)
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        x_set = self.__scale(data=x_set)
        y_hat = self.__model.predict(x_set)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        x_set = self.__scale(x_set)
        score = self.__model.score(x_set, y_set)
        return score

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        """reshape data set"""
        if len(data.shape) == 1:
            data = np.reshape(data, newshape=(data.shape[0], 1))
            data = np.asarray(data)
        scalar = StandardScaler()
        scalar.fit(data)
        data = scalar.transform(data)
        return data


class ParallelSupervisedClassifier(object):

    def __init__(self, model_0: SupervisedClassifier, model_1: SupervisedClassifier):
        """initializing parallel supervised classifier"""
        self.__model_0 = model_0
        self.__model_1 = model_1
        return

    def fit(self, x_set: np.ndarray, y_set: np.ndarray):
        """training learner"""
        self.__model_0.fit(x_set[:, 0], y_set[:, 0])
        self.__model_1.fit(x_set[:, 1], y_set[:, 1])
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        y_0 = self.__model_0.predict(x_set[:, 0])
        y_1 = self.__model_1.predict(x_set[:, 1])
        y_hat = np.stack(arrays=(y_0, y_1), axis=1)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def error_bitwise(self, x_set, y_set) -> float:
        """mean error on prediction in each dimension"""
        err = 1 - self.accuracy_bitwise(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum((y_hat[:, 0] == y_set[:, 0]) & (y_hat[:, 1] == y_set[:, 1]))
        score = float(score) / float(y_set.shape[0])
        return score

    def accuracy_bitwise(self, x_set, y_set) -> float:
        """mean accuracy on prediction in each dimension"""
        y_hat = self.predict(x_set)
        score_0 = np.sum(y_hat[:, 0] == y_set[:, 0])
        score_1 = np.sum(y_hat[:, 1] == y_set[:, 1])
        return float(score_0 + score_1) / 2.00

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class UnsupervisedClassifier(object):

    def __init__(self, model):
        """initializing supervised classifier"""
        self.__model = model
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        x_set = self.__scale(data=x_set)
        y_hat = self.__model.fit_predict(x_set)
        centers = self.__find_centers()
        y_temp = np.zeros(shape=y_hat.shape)
        for index in range(0, len(centers)):
            y_temp = np.where(y_hat == index, centers[index], y_temp)
        y_hat = np.asarray(y_temp)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_hat == y_set)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix

    def __find_centers(self) -> np.ndarray:
        """centering clusters"""
        c_0 = -1
        c_1 = +1
        res = np.asarray((c_0, c_1))
        centers = self.__get_centers()
        if centers[0] > centers[1]:
            res[0] = c_1
            res[1] = c_0
        return res

    def get_iter(self) -> int:
        """getting number of iterations"""
        return self.__model.n_iter_

    def __get_centers(self) -> np.ndarray:
        """getting centers of model"""
        raise Exception('not implemented!')

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        """reshape data set"""
        if len(data.shape) == 1:
            data = np.reshape(data, newshape=(data.shape[0], 1))
            data = np.asarray(data)
        scalar = StandardScaler()
        scalar.fit(data)
        data = scalar.transform(data)
        return data


class ParallelUnsupervisedClassifier(object):

    def __init__(self, model_0, model_1):
        """initializing parallel unsupervised classifier"""
        self.__model_0 = model_0
        self.__model_1 = model_1
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        y_0 = self.__model_0.predict(x_set[:, 0])
        y_1 = self.__model_1.predict(x_set[:, 1])
        y_hat = np.stack(arrays=(y_0, y_1), axis=1)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def error_bitwise(self, x_set, y_set) -> float:
        """mean error on prediction in each dimension"""
        err = 1 - self.accuracy_bitwise(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum((y_hat[:, 0] == y_set[:, 0]) & (y_hat[:, 1] == y_set[:, 1]))
        score = float(score) / float(y_set.shape[0])
        return score

    def accuracy_bitwise(self, x_set, y_set) -> float:
        """mean accuracy on prediction in each dimension"""
        y_hat = self.predict(x_set)
        score_0 = np.sum(y_hat[:, 0] == y_set[:, 0])
        score_1 = np.sum(y_hat[:, 1] == y_set[:, 1])
        return float(score_0 + score_1) / 2.00

    def get_iter(self) -> int:
        """getting number of iterations"""
        iter_0 = self.__model_0.get_iter()
        iter_1 = self.__model_1.get_iter()
        return iter_0 + iter_1

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class BinaryPredictor:

    def __init__(self, line=0.0):
        """initializing the predictor"""
        self.__line = line
        return

    def predict(self, x_set) -> np.ndarray:
        """performing prediction"""
        y_predicted = np.ones(x_set.shape)
        y_predicted = np.where(x_set < self.__line, -1, y_predicted)
        return y_predicted

    def error(self, x_set, y_set) -> float:
        """error of prediction"""
        return 1 - self.accuracy(x_set, y_set)

    def accuracy(self, x_set, y_set) -> float:
        """accuracy of prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_set == y_hat)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set, y_set) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class TernaryPredictor:

    def __init__(self, line=(-0.36, 0.37), gain=1.00):
        """initializing the predictor"""
        self.__line = [gain * temp for temp in line]
        return

    def predict(self, x_set) -> np.ndarray:
        """performing prediction"""
        y_predicted = np.zeros(x_set.shape)
        y_predicted = np.where(x_set < self.__line[0], -1, y_predicted)
        y_predicted = np.where(x_set >= self.__line[1], 1, y_predicted)
        return y_predicted

    def error(self, x_set, y_set) -> float:
        """error of prediction"""
        return 1 - self.accuracy(x_set, y_set)

    def accuracy(self, x_set, y_set) -> float:
        """accuracy of prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_set == y_hat)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set, y_set) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class QuadraturePredictor:

    def __init__(self, line=0.0):
        """initializing the predictor"""
        self.__line = line
        return

    def predict(self, x_set) -> np.ndarray:
        """performing prediction"""
        y_predicted = np.ones(x_set.shape)
        y_predicted = np.where(x_set < self.__line, -1, y_predicted)
        return y_predicted

    def error(self, x_set, y_set) -> float:
        """error of prediction"""
        return 1 - self.accuracy(x_set, y_set)

    def accuracy(self, x_set, y_set) -> float:
        """accuracy of prediction"""
        y_hat = self.predict(x_set)
        score = np.sum((y_set[:, 0] == y_hat[:, 0]) & (y_set[:, 1] == y_hat[:, 1]))
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set, y_set) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class SixteenPredictor:

    def __init__(self, line=(-2.00, 0.00, 2.00), gain=1.00):
        """initializing the predictor"""
        self.__line = [gain * temp for temp in line]
        return

    def predict(self, x_set) -> np.ndarray:
        """performing prediction"""
        y_predicted = np.full(x_set.shape, 3)
        y_predicted = np.where((x_set < self.__line[0]), -3, y_predicted)
        y_predicted = np.where((x_set < self.__line[1]) & (x_set >= self.__line[0]), -1, y_predicted)
        y_predicted = np.where((x_set < self.__line[2]) & (x_set >= self.__line[1]), +1, y_predicted)
        y_predicted = np.where(x_set >= self.__line[2], +3, y_predicted)
        return y_predicted

    def error(self, x_set, y_set) -> float:
        """error of prediction"""
        return 1 - self.accuracy(x_set, y_set)

    def accuracy(self, x_set, y_set) -> float:
        """accuracy of prediction"""
        y_hat = self.predict(x_set)
        score = np.sum((y_set[:, 0] == y_hat[:, 0]) & (y_set[:, 1] == y_hat[:, 1]))
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set, y_set) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix


class BayesClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing Gaussian naive Bayes classifier"""
        model = GaussianNB()
        super().__init__(model=model)
        return


class LogisticClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing logistic regression classifier"""
        model = LogisticRegression(solver='lbfgs', multi_class='auto')
        super().__init__(model=model)
        return


class TreeClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing decision tree classifier"""
        model = DecisionTreeClassifier(max_depth=10)
        super().__init__(model=model)
        return


class ForestClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing random forests classifier"""
        model = RandomForestClassifier(n_estimators=5, max_depth=10)
        super().__init__(model=model)
        return


class VectorClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing support vector machines classifier"""
        model = SVC(kernel='rbf', probability=True, gamma='auto')
        super().__init__(model=model)
        return


class NearestClassifier(SupervisedClassifier):

    def __init__(self):
        """initializing nearest neighbors classifier"""
        model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        super().__init__(model=model)
        return


class ParallelBayesClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel Gaussian naive Bayes classifier"""
        model_0 = BayesClassifier()
        model_1 = BayesClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelLogisticClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel logistic classifier"""
        model_0 = LogisticClassifier()
        model_1 = LogisticClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelTreeClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel decision tree classifier"""
        model_0 = TreeClassifier()
        model_1 = TreeClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelForestClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel random forests classifier"""
        model_0 = ForestClassifier()
        model_1 = ForestClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelVectorClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel support vector machines classifier"""
        model_0 = VectorClassifier()
        model_1 = VectorClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelNearestClassifier(ParallelSupervisedClassifier):

    def __init__(self):
        """initializing parallel nearest neighbors classifier"""
        model_0 = NearestClassifier()
        model_1 = NearestClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class MeansClusterClassifier:

    def __init__(self):
        """initializing means clustering classifier"""
        self.__model = KMeans(n_clusters=2)
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        x_set = self.__scale(data=x_set)
        y_hat = self.__model.fit_predict(x_set)
        centers = self.__find_centers()
        y_temp = np.zeros(shape=y_hat.shape)
        for index in range(0, len(centers)):
            y_temp = np.where(y_hat == index, centers[index], y_temp)
        y_hat = np.asarray(y_temp)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_hat == y_set)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix

    def __find_centers(self) -> np.ndarray:
        """centering clusters"""
        c_0 = -1
        c_1 = +1
        res = np.asarray((c_0, c_1))
        centers = self.__get_centers()
        if centers[0] > centers[1]:
            res[0] = c_1
            res[1] = c_0
        return res

    def get_iter(self) -> int:
        """getting number of iterations"""
        return self.__model.n_iter_

    def __get_centers(self) -> np.ndarray:
        """getting centers"""
        centers = np.asarray(self.__model.cluster_centers_)
        return centers

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        """reshape data set"""
        if len(data.shape) == 1:
            data = np.reshape(data, newshape=(data.shape[0], 1))
        data = np.asarray(data)
        return data


class AgglomerativeClusterClassifier:

    def __init__(self):
        """initializing agglomerative clustering classifier"""
        self.__model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None)
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        x_set = self.__scale(data=x_set)
        y_hat = self.__model.fit_predict(x_set)
        centers = self.__find_centers(x_set)
        y_temp = np.zeros(shape=y_hat.shape)
        for index in range(0, len(centers)):
            y_temp = np.where(y_hat == index, centers[index], y_temp)
        y_hat = np.asarray(y_temp)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_hat == y_set)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix

    def __find_centers(self, x_set: np.ndarray) -> np.ndarray:
        """centering clusters"""
        c_0 = -1
        c_1 = +1
        res = np.asarray((c_0, c_1))
        centers = self.__get_centers(x_set)
        if centers[0] > centers[1]:
            res[0] = c_1
            res[1] = c_0
        return res

    def get_iter(self) -> int:
        """getting number of connected components"""
        return self.__model.n_connected_components_

    def __get_centers(self, x_set: np.ndarray) -> np.ndarray:
        """getting centers"""
        res = np.zeros(shape=2)
        for index in range(0, self.__model.n_clusters_):
            temp = np.where(self.__model.labels_ == index)
            res[index] = np.mean(x_set[temp])
        return res

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        """reshape data set"""
        if len(data.shape) == 1:
            data = np.reshape(data, newshape=(data.shape[0], 1))
        data = np.asarray(data)
        return data


class GaussianMixtureClassifier:

    def __init__(self):
        """initializing Gaussian mixture classifier"""
        self.__model = GaussianMixture(n_components=2)
        return

    def predict(self, x_set: np.ndarray) -> np.ndarray:
        """predicting future data"""
        x_set = self.__scale(data=x_set)
        y_hat = self.__model.fit_predict(x_set)
        centers = self.__find_centers()
        y_temp = np.zeros(shape=y_hat.shape)
        for index in range(0, len(centers)):
            y_temp = np.where(y_hat == index, centers[index], y_temp)
        y_hat = np.asarray(y_temp)
        return y_hat

    def error(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """error on prediction"""
        err = 1 - self.accuracy(x_set, y_set)
        return err

    def accuracy(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        """accuracy on prediction"""
        y_hat = self.predict(x_set)
        score = np.sum(y_hat == y_set)
        score = float(score) / float(y_set.shape[0])
        return score

    def confusion(self, x_set: np.ndarray, y_set: np.ndarray) -> np.ndarray:
        """confusion matrix"""
        y_hat = self.predict(x_set)
        matrix = confusion_matrix(y_set, y_hat)
        return matrix

    def __find_centers(self) -> np.ndarray:
        """centering clusters"""
        c_0 = -1
        c_1 = +1
        res = np.asarray((c_0, c_1))
        centers = self.__get_centers()
        if centers[0] > centers[1]:
            res[0] = c_1
            res[1] = c_0
        return res

    def get_iter(self) -> int:
        """getting number of iterations"""
        return self.__model.n_iter_

    def __get_centers(self) -> np.ndarray:
        """getting centers"""
        centers = np.asarray(self.__model.means_)
        return centers

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        """reshape data set"""
        if len(data.shape) == 1:
            data = np.reshape(data, newshape=(data.shape[0], 1))
        data = np.asarray(data)
        return data


class ParallelMeansClusterClassifier(ParallelUnsupervisedClassifier):

    def __init__(self):
        """initializing mean clustering classifier"""
        model_0 = MeansClusterClassifier()
        model_1 = MeansClusterClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelAgglomerativeClusterClassifier(ParallelUnsupervisedClassifier):

    def __init__(self):
        """initializing agglomerative clustering classifier"""
        model_0 = AgglomerativeClusterClassifier()
        model_1 = AgglomerativeClusterClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return


class ParallelGaussianMixtureClassifier(ParallelUnsupervisedClassifier):

    def __init__(self):
        """initializing Gaussian mixture classifier"""
        model_0 = GaussianMixtureClassifier()
        model_1 = GaussianMixtureClassifier()
        super().__init__(model_0=model_0, model_1=model_1)
        return
