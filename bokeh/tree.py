import matplotlib.pyplot as plt
import networkx as nx
import json
import pandas as pd
import re

from functools import partial
from random import random
from bokeh.layouts import column
from bokeh.models import Button, Slider, Plot, Range1d, MultiLine, Circle, Span, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool, StaticLayoutProvider, ColumnDataSource, Grid, LinearAxis, Segment
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import RdYlBu3, Spectral4, d3
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_file

from threading import Thread
from functools import partial
from time import sleep

points = pd.DataFrame(pd.read_csv("datasets/gaussian/hundred.csv", delimiter=","))

def file_exists(path):
    exists = False
    f = None
    try:
        f = open(path)
        # Do something with the file
        exists = True
    except IOError:
        print("File '{}' not found.".format(path))
    finally:
        if not f is None:
            f.close()
    return exists

def replot(i):
    path = "tree/{}.gml".format(i)
    with open(path) as json_file:
        data = json.load(json_file)
    G = nx.readwrite.json_graph.node_link_graph(data)
    
    title = "GOSDT Tree (iteration = {})".format(i)
    plot = figure(title=title, x_axis_label='Threshold', y_axis_label='Support')

    spans = []
    sources = []

    colors = []
    x = []
    y0 = []
    y1 = []
    for node in G:
        if not 'scores' in G.nodes[node] or len(G.nodes[node]['scores']) == 0:
            continue

        temperature = []
        for threshold, upperbound in G.nodes[node]['scores'].items():
            x.append(float(threshold))
            y1.append(G.nodes[node]['support'])
            if threshold == G.nodes[node]['threshold']:
                y0.append(0.0)
            else:
                y0.append(G.nodes[node]['support'] - 0.02)
            temperature.append(upperbound)

        min_temp = min(temperature)
        max_temp = max(temperature)

        if min_temp < max_temp:
            temperature = [ round(15 * (t - min_temp) / (max_temp - min_temp)) for t in temperature ]
        else:
            temperature = [ round(7.5) for t in temperature ]


        for t in temperature:
            if t == 0:
                colors.append('#FF5500')
            elif t == 1:
                colors.append('#EE5511')
            elif t == 2:
                colors.append('#DD5522')
            elif t == 3:
                colors.append('#CC5533')
            elif t == 4:
                colors.append('#BB5544')
            elif t == 5:
                colors.append('#AA5555')
            elif t == 6:
                colors.append('#995566')
            elif t == 7:
                colors.append('#885577')
            elif t == 8:
                colors.append('#775588')
            elif t == 9:
                colors.append('#665599')
            elif t == 10:
                colors.append('#5555AA')
            elif t == 11:
                colors.append('#4455BB')
            elif t == 12:
                colors.append('#3355CC')
            elif t == 13:
                colors.append('#2255DD')
            elif t == 14:
                colors.append('#1155EE')
            elif t == 15:
                colors.append('#0055FF')
            
    source = ColumnDataSource(dict(
            x=x,
            y0=y0,
            y1=y1,
            color=colors
        )
    )
    glyph = Segment(x0='x', y0='y0', x1='x', y1='y1', line_color="color", line_width=3, line_alpha=0.4)
    plot.add_glyph(source, glyph)

    plot.circle(points.values[:,0], points.values[:,1], legend_label='ground truth', fill_alpha=0.2, size=3)

    return plot

i = 1
min_iter = 0
while not file_exists("tree/{}.gml".format(min_iter)):
    min_iter += 1
max_iter = min_iter
while file_exists("tree/{}.gml".format(max_iter + 1)):
    max_iter += 1

doc = curdoc()

iterator = Slider(title="Iteration", value=min_iter, start=min_iter, end=max_iter, step=1)
layout = column(iterator, replot(iterator.value))

def render(attrname, old, new):
    layout.children[1] = replot(iterator.value)

iterator.on_change('value', render)
doc.add_root(layout)