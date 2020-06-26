import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def file_exists(path):
    exists = False
    file_descriptor = None
    try:
        file_descriptor = open(path)
        # Do something with the file
        exists = True
    except IOError:
        # print("File '{}' not found.".format(path))
        pass
    finally:
        if not file_descriptor is None:
            file_descriptor.close()
    return exists

# Python 3.4
# matplotlib 1.5.3
class Radar(object):
    def __init__(self, fig, titles, labels, rotation=0, rect=None):
        if rect is None:
            rect = [0.01, 0.05, 0.99, 0.95]

        self.n = len(titles) # Number of dimensions
        self.angles = np.arange(0, 360, 360.0 / self.n)
        self.axes = [ fig.add_axes(rect, projection="polar", label="axes%d" % i) for i in range(self.n) ]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=8)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, len(labels[0]) + 1), angle=angle, labels=label, fontsize=5, alpha=0.5)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, len(labels[0]) + 1)
            ax.set_theta_offset(np.deg2rad(rotation))

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)

def plot_radar(dataset, algorithms):
    configuration_file_path = "python/configurations/configuration.json"
    if not file_exists(configuration_file_path):
        exit(1)
    with open(configuration_file_path, 'r') as configuration_source:
        configuration = configuration_source.read()
        configuration = json.loads(configuration)
    
    dataframe = None
    regularizations = configuration["regularization"]
    list.reverse(regularizations)
    for regularization in regularizations:
        for algorithm in algorithms:
            data_file_path = "results/benchmark/trials_{}_{}_{}.csv".format(dataset, algorithm, regularization)
            if not file_exists(data_file_path):
                exit(1)
            trials = pd.read_csv(data_file_path)
            if dataframe is None:
                dataframe = trials
            else:
                dataframe = dataframe.append(trials, ignore_index=True)

    max_depth = dataframe['depth'].max()
    max_leaves = dataframe['leaves'].max()
    max_nodes = dataframe['nodes'].max()
    max_time = dataframe['time'].max()

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    fig.suptitle("Tree Classification Benchmark (Dataset = {})".format(dataset), fontsize=12)

    ticks = 4
    titles = ['train err.', 'test err.', 'depth', 'leaves', 'nodes', 'time']
    labels = [
        [ round(i, 1) for  i in np.linspace(0.0, 1.0, num=ticks) ],
        [ round(i, 1) for  i in np.linspace(0.0, 1.0, num=ticks) ],
        [ round(i, 1) for  i in np.linspace(0.0, 1.0 * max_depth, num=ticks) ],
        [ round(i, 1) for  i in np.linspace(0.0, 1.0 * max_leaves, num=ticks) ],
        [ round(i, 1) for  i in np.linspace(0.0, 1.0 * max_nodes, num=ticks) ],
        [ round(i, 1) for  i in np.linspace(0.0, 1.0 * max_time, num=ticks) ],
    ]

    colors = ['c', 'm', 'y', 'k', 'b', 'g', 'r']
    columns = len(algorithms) + 1
    rows = len(regularizations)
    outer_frame = [0.01, 0.01, 0.99, 0.95]
    header_space = [0.05, 0.025]
    padding = 0.035
    width = (outer_frame[2] - outer_frame[0] - (columns + 1) * padding - header_space[0]) / columns
    height = (outer_frame[3] - outer_frame[1] - (rows + 1) * padding - header_space[1])  / rows
    for i, regularization in enumerate(regularizations):
        subtitle = r'$\lambda = $' + str(regularization)
        fig.text(
            outer_frame[0],
            outer_frame[1] + padding + i * (height + padding) + 0.5 * height,
            subtitle, fontsize=10, va='center')
    for j, algorithm in enumerate(algorithms + ['all']):
        subtitle = algorithm
        fig.text(
            outer_frame[0] + header_space[0] + padding + j * (width + padding) + 0.5 * width,
            outer_frame[1] + padding + rows * (height + padding),
            subtitle, fontsize=10, ha='center', va='center')
    for i, regularization in enumerate(regularizations):
        for j, algorithm in enumerate(algorithms):
            frame = [
                outer_frame[0] + header_space[0] + padding + j * (width + padding),
                outer_frame[1] + padding + i * (height + padding),
                width,
                height
            ]
            radar = Radar(fig, titles, labels, rect=frame)
            subset = dataframe.copy()
            subset = subset[subset['regularization']==regularization]
            subset = subset[subset['algorithm']==algorithm]
            (n, m) = subset.shape
            for k in range(n):
                point = [
                    subset.values[k, 3],
                    subset.values[k, 4],
                    subset.values[k, 5] / max_depth,
                    subset.values[k, 6] / max_leaves,
                    subset.values[k, 7] / max_nodes,
                    subset.values[k, 8] / max_time
                ]
                point = [ 1.0 + ticks * v for v in point ]
                radar.plot(point, '-', lw=1, color=colors[j], alpha=0.2)

    for i, regularization in enumerate(regularizations):
        frame = [
                outer_frame[0] + header_space[0] + padding + (columns - 1) * (width + padding),
                outer_frame[1] + padding + i * (height + padding),
                width,
                height
            ]
        radar = Radar(fig, titles, labels, rect=frame)
        for j, algorithm in enumerate(algorithms):
            
            subset = dataframe.copy()
            subset = subset[subset['regularization']==regularization]
            subset = subset[subset['algorithm']==algorithm]
            (n, m) = subset.shape
            for k in range(n):
                point = [
                    subset.values[k, 3],
                    subset.values[k, 4],
                    subset.values[k, 5] / max_depth,
                    subset.values[k, 6] / max_leaves,
                    subset.values[k, 7] / max_nodes,
                    subset.values[k, 8] / max_time
                ]
                point = [ 1.0 + ticks * v for v in point ]
                radar.plot(point, '-', lw=1, color=colors[j], alpha=0.2)
    plt.show()
