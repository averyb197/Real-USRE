import matplotlib.pyplot as plt
import seaborn as sns

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

