
import numpy as np
import joblib as jb
import pandas as pd
import csv as csv
import xlwt as xl
import seaborn as sb

from matplotlib import pyplot as plt

from pandas import DataFrame
from matplotlib.ticker import FuncFormatter as funcFormat
from matplotlib.ticker import MultipleLocator


root_path = '/' # change it for yourself


class DataReader:

    def __init__(self):
        self.path = root_path + '/out/data/'
        self.folder = root_path + '/out/model/'
        return

    def read_data(self, data_name: str, rows=None) -> DataFrame:
        """loading data"""
        try:
            data = pd.read_csv(self.path + data_name + '.csv', nrows=rows)
            return data
        except FileNotFoundError:
            raise Exception('data not found!')

    def read_model(self, model_name: str):
        """loading the saved model from the disk"""
        model = jb.load(self.folder + model_name + '.sav')
        return model


class DataWriter:

    def __init__(self):
        self.path = root_path + '/out/data/'
        self.folder = root_path + '/out/model/'
        self.count = 0
        return

    def data_to_csv(self, usage: list, time: list, user_list: list, symbol_list: list, desired_list: list,
                    received_list: list, file_name: str):
        """saving specific data to a file for further training"""
        data = {'channel': usage, 'time': time, 'userID': user_list, 'symbol': symbol_list, 'desired': desired_list,
                'received': received_list}
        df = DataFrame(data, columns=('channel', 'time', 'userID', 'symbol', 'desired', 'received'))
        if self.count == 0:
            with open(self.path + file_name + '.csv', 'w') as f:
                df.to_csv(f, header=True, index=None)
        else:
            with open(self.path + file_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=None)
        self.count += 1
        return

    def comment_csv(self, text: str, name: str):
        """creating a comment file for more information"""
        comment = [text]
        with open(self.path + name + '_info.csv', 'w') as f_comment:
            writer = csv.writer(f_comment, delimiter=' ')
            writer.writerow(comment)
        f_comment.close()
        return

    def frame_to_csv(self, data_frame: DataFrame, name: str):
        """saving a data frame on disk"""
        data_frame.to_csv(self.path + name + '.csv', index=False)
        return

    def save_model(self, model, model_name: str):
        """saving the trained model on disk"""
        jb.dump(model, self.folder + model_name + '.sav')
        return

    def data_to_xl(self, workbook: xl.Workbook(), data_name: str):
        """saving data work book on disk"""
        temp_wb = workbook
        temp_wb.save(self.path + data_name + '.xls')
        return


class Plotter:

    def __init__(self, grid=False, dark=False, x_axis=None, y_axis=None, x_title=None, y_title=None, g_plot=False):
        """initializing the plotter"""
        self.path = root_path + '/out/figure/'
        self.fig = plt
        self.map = sb
        self.dark = dark
        if dark:
            plt.style.use('dark_background')
        elif g_plot:
            plt.style.use('ggplot')
        if x_title is not None:
            plt.xlabel(x_title)
        if y_title is not None:
            plt.ylabel(y_title)
        if x_axis is not None:
            plt.xticks(ticks=x_axis)
            plt.xlim(x_axis[0], x_axis[-1])
        if y_axis is not None:
            plt.yticks(ticks=y_axis)
            plt.ylim(y_axis[0], y_axis[-1])
        if grid:
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', axis='both', color=[0.75, 0.75, 0.75])
            plt.grid(which='minor', linestyle=':', axis='both', color=[0.9, 0.9, 0.9])
        return

    def set_limit(self, x_limit=None, y_limit=None):
        """external method to set boundaries"""
        if x_limit is not None:
            self.fig.xlim(x_limit[0], x_limit[-1])
        if y_limit is not None:
            self.fig.ylim(y_limit[0], y_limit[-1])
        return

    def set_title(self, x_label=None, y_label=None):
        """external method to set labels"""
        self.fig.xlabel(x_label)
        self.fig.ylabel(y_label)
        return

    def to_linear(self, x, y, color=None, marker='', style='-', label=None):
        """linear plotting function"""
        self.fig.plot(x, y, color=color, linestyle=style, marker=marker, label=label)
        if label is not None:
            plt.legend()
        return

    def to_log(self, x, y, color=None, marker='', style='-', label=None):
        """log plotting function"""
        self.fig.semilogy(x, y, color=color, linewidth='2', linestyle=style, marker=marker, label=label, markevery=0.1)
        if label is not None:
            plt.legend()
        return

    def to_scatter(self, x, y, edge_color=None, in_color=None, marker=None, label=None, alpha=0.8):
        """scatter plotting function"""
        if in_color is None:
            if self.dark:
                in_color = 'black'
            else:
                in_color = 'white'
        if edge_color is None:
            if self.dark:
                edge_color = 'white'
            else:
                edge_color = 'black'
        self.fig.scatter(x, y, marker=marker, label=label, c=in_color, edgecolors=edge_color, alpha=alpha)
        if label is not None:
            plt.legend()
        return

    def to_correlation(self, data_frame: DataFrame):
        """visualizing the correlation matrix of the data"""
        c = data_frame.corr()
        self.map.heatmap(c, xticklabels=c.columns, yticklabels=c.columns, cbar=True, vmin=0, vmax=1, cmap='Reds')
        return

    def fast_plot(self, x, y, label=None):
        """for test cases: see a simple figure of data"""
        self.fig.plot(x, y, label=label)
        if label is not None:
            self.fig.legend()
        return

    def fast_scatter(self, x, y, label=None, alpha=1.0):
        """for test cases: see a simple scatter plot"""
        self.fig.scatter(x, y, label=label, alpha=alpha)
        if label is not None:
            self.fig.legend()
        return

    def to_radian(self, x, y):
        """simple figure based on numpy.pi"""
        self.fig.plot(x, y)
        ax = plt.gca()
        temp = '{:.0g}$\pi$'
        ax.xaxis.set_major_formatter(funcFormat(lambda v, p: temp.format(v / np.pi) if v != 0 else '0'))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 4))
        return

    def save(self, figure_name: str):
        """save the current figure"""
        try:
            plt.savefig(self.path + figure_name)
            plt.close()
        except NotADirectoryError:
            raise Exception('problem with the path!')

    def show(self):
        """show the current figure"""
        self.fig.show()
        return True

    def clear(self):
        """clear all running visualizer"""
        self.fig.close()
        return True
