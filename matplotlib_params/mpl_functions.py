import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def plot_settings(defaut_settings = True):

    if defaut_settings :
        plt.style.use(['default'])
    
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['font.size'] = 14  # global font size

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['axes.formatter.limits'] = -2,3
    mpl.rcParams['grid.linewidth'] = 0.4
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.numpoints'] = 5
    mpl.rcParams['legend.scatterpoints'] = 5
    mpl.rcParams['legend.edgecolor'] = 'gray'
    mpl.rcParams['patch.linewidth'] = 1

    mpl.rcParams['legend.borderpad'] = 0.4
    mpl.rcParams['legend.labelspacing'] = 0.5

    mpl.rcParams['legend.handlelength'] = 2.4
    mpl.rcParams['legend.handleheight'] = 0.6
    mpl.rcParams['legend.handletextpad'] = 0.8
    mpl.rcParams['legend.borderaxespad'] = 0.5
    mpl.rcParams['legend.columnspacing'] = 2

    mpl.rcParams['axes.edgecolor'] = 'gray'
    mpl.rcParams['axes.linewidth'] = 0.8

def custom_legend(ax, *args, **kwargs):
    handler_map = {}
    for line in ax.get_lines():
        linestyle = line.get_linestyle()
        if linestyle in ['--', '-.', ':']: 
            handler_map[line] = HandlerLine2D(numpoints=2)

    kwargs.setdefault('handler_map', handler_map)
    kwargs.setdefault('handlelength', 2.7)
    return ax.legend(*args, **kwargs)