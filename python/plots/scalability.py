import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import log
import pandas as pd
import numpy as np
import seaborn as sns
import re

def plot_scalability(dataset, algorithms):
    plt.rc('font', size=18)
    colors = ['#509eba', '#e3a042', '#d77ede', '#233c82', '#613717', '#16baea']
    markers = ['o', 's', 'D', 'v', '^', '*', 'o', 's', 'D', 'v', '^']

    time_limit = 300


    for x_axis, prefix, x_label in [('binary_features', 'features', 'Number of Features'), ('samples', 'samples', 'Number of Samples')]:
        for y_axis, y_label in [('training time', 'Time'), ('training time', 'Slow-Down')]:
            for scale in ['regular', 'small']:
                plt.figure(figsize=(8, 5), dpi=80)

                x_max = 0

                for j, algorithm in enumerate(sorted(set(algorithms))):
                    data = pd.DataFrame(pd.read_csv('results/scalability/{}/{}_{}_{}.csv'.format(prefix, prefix, dataset, algorithm)))
                    if data[data['time'] > 0][data['time'] < time_limit].shape[0] == 0:
                        continue
                    
                    if algorithm != 'cart':
                        x_max = max(x_max, max(data[data['time'] > 0][data['time'] < time_limit][x_axis]))

                    x = data[x_axis][data['time'] > 0][data['time'] < time_limit]
                    
                    if y_label == "Slow-Down": # Slow-Down
                        base_time = min(data['time'][data['time'] > 0][data['time'] < time_limit])
                        y = data[data['time'] > 0][data['time'] < time_limit]['time'].apply(lambda x : x / base_time)
                        plt.errorbar(x, y, label=algorithm,
                        markersize=5, marker=markers[j], 
                        color=colors[j], alpha=0.75,
                        linewidth=0.5, linestyle='none')                    
                    elif y_label == "Time": # Runtime
                        y = data[data['time'] > 0][data['time'] < time_limit]['time']
                        plt.errorbar(x, y, label=algorithm,
                            markersize=5, marker=markers[j], 
                            color=colors[j], alpha=0.6,
                            linewidth=0.5, linestyle='none')

                # plt.yscale('log', basey=10)

                plt.legend()
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.xlim(-x_max*0.01, x_max * 1.5)
                plt.legend()
                plt.title("{} vs {}\n({})".format(y_label, x_label, dataset))

                if scale == "small":
                    plt.ylim(-5, 100)
                plt.savefig("figures/scalability/{}_{}_{}_{}".format(y_label, x_label, dataset, scale).replace(" ", "_").lower(), bbox_inches='tight')