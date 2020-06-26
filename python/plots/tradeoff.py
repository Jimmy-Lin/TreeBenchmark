import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits import mplot3d
from math import log
import pandas as pd
import numpy as np
import seaborn as sns
import re
import json

def file_exists(path):
    exists = False
    file_descriptor = None
    try:
        file_descriptor = open(path)
        exists = True
    except IOError:
        print("File '{}' not found.".format(path))
        pass
    finally:
        if not file_descriptor is None:
            file_descriptor.close()
    return exists

def plot_tradeoff(dataset, algorithms):
    for y_axis in ['Training Accuracy', 'Test Accuracy']:
        configuration_file_path = "python/configurations/configurations.json"
        if not file_exists(configuration_file_path):
            exit(1)
        with open(configuration_file_path, 'r') as configuration_source:
            configuration = configuration_source.read()
            configuration = json.loads(configuration)

        # Styling 
        plt.rc('font', size = 18)
        colors = ['#509eba', '#e3a042', '#d77ede', '#233c82', '#613717']
        markers = ['o', 's', 'D', 'v', '^', '*']

        # Axis Labels
        x_axis = '# Leaves'

        # Resolution + Size
        plt.figure(figsize=(8, 5), dpi=80)

        # Load data
        dataframe = None
        for i, algorithm in enumerate(sorted(set(algorithms))): # Go through algorithms in alphabetic order
            data_file_path = "results/tradeoff/{}_{}.csv".format(dataset, algorithm)
            if not file_exists(data_file_path):
                exit(1)
            trials = pd.read_csv(data_file_path)
            if dataframe is None:
                dataframe = trials
            else:
                dataframe = dataframe.append(trials, ignore_index=True)

        # Further filtering
        dataframe = dataframe[dataframe['Training Time'] > 0] # Don't show crashes
        dataframe = dataframe[dataframe['Training Time'] < 300] # Don't show time-outs
        dataframe = dataframe[dataframe['# Leaves'] <= 32] # Don't show extremely large trees

        x_max = 0

        # Go through algorithms in alphabetic order, plot each algorithm's curve
        for i, algorithm in enumerate(sorted(set(algorithms))):
            data = dataframe[dataframe['Algorithm'] == algorithm]

            x = []; y = []
            x_low = []; x_high = []
            y_low = []; y_high = []
            errorboxes = []
            points = set()

            # Iterate through each unique configuration
            for depth, width, reg in sorted(set(tuple(row.tolist()) for row in data[['Depth Limit', 'Width Limit', 'Regularization']].values)):
                point_results = data[data['Depth Limit'] == depth][data['Width Limit'] == width][data['Regularization'] == reg]

                if len(point_results) <= 2:
                    continue

                x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
                y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])


                point = (
                    x_iqr[0.5], y_iqr[0.5],
                    x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
                    y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

                if not point in points:
                    points.add(point)
                    x.append(point[0]); y.append(point[1])
                    x_low.append(point[2]); x_high.append(point[3])
                    y_low.append(point[4]); y_high.append(point[5])

            for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
                rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
                errorboxes.append(rect)

            if algorithm == 'gosdt':
                x_max = max(x)

            # # Create patch collection with specified colour/alpha
            # pc = PatchCollection(errorboxes, facecolor=colors[i], edgecolor=colors[i], alpha=0.0)
            # ax.add_collection(pc)
            
            plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high], label=algorithm,
                markersize=5, marker=markers[i], 
                color=colors[i], alpha=0.75,
                linewidth=1, linestyle='none')

        # plt.xscale('log')
        # plt.margins(10, 10)
        # plt.xlim(0, min(x_max + 1, 100))

        plt.xlabel('Number of Leaves')
        plt.ylabel(y_axis + " (%)")
        if dataset == 'monk-3':
            plt.legend(loc='lower right')
        
        plt.title("{} vs Number of Leaves\n({})".format(y_axis, dataset))
        plt.savefig("figures/tradeoff/{}_vs_leaves_{}.png".format(y_axis.lower().replace(' ', '_'), dataset), bbox_inches='tight')


def split_tradeoff(dataset, algorithms):
    data_file_path = "results/tradeoff/{}.csv".format(dataset)
    if not file_exists(data_file_path):
        exit(1)
    dataframe = pd.read_csv(data_file_path)

    for i, algorithm in enumerate(sorted(set(algorithms))):
        data = dataframe[dataframe['Algorithm'] == algorithm]
        output_path = "results/tradeoff/{}_{}.csv".format(dataset, algorithm)
        data.to_csv(output_path, index= False)
