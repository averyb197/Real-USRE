import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
def make_pretty():
    sns.set_theme(palette="plasma")
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['savefig.edgecolor'] = 'black'

def colorline(X, Y, t, cmap="plasma"):
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    normed_t = Normalize(vmin=t.min(), vmax=t.max())
    line_collection = LineCollection(segments, array=t, norm=normed_t, cmap=cmap)

    return line_collection

