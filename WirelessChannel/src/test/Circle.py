
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def generate(mu):
    """given the range, generate the number of users accordingly"""
    factor = 40 / numpy.pi
    users = int(mu * factor)
    diff = numpy.linspace(-1 * mu, mu, users, dtype='float')
    return diff


def draw():
    """testing"""
    lim = numpy.pi
    m = generate(lim)
    plt.plot(m, numpy.sin(m))
    plt.plot(m, numpy.cos(m))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: '{:.0g}$\pi$'.format(val / numpy.pi) if val != 0 else '0'
    ))
    ax.xaxis.set_major_locator(MultipleLocator(base=numpy.pi / 4))
    plt.show()
    print(len(m))
    return
