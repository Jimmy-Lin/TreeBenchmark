import matplotlib.pyplot as plt
import networkx as nx
import json

from functools import partial
from random import random
from bokeh.layouts import column
from bokeh.models import Button, Slider, Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool, StaticLayoutProvider
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import RdYlBu3, Spectral4, d3
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_file

from threading import Thread
from functools import partial
from time import sleep

def file_exists(path):
    exists = False
    f = None
    try:
        f = open(path)
        exists = True
    except IOError:
        print("File '{}' not found.".format(path))
    finally:
        if not f is None:
            f.close()
    return exists

def replot(i):
    plot = Plot(plot_width=1400, plot_height=800,
                x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    plot.title.text = "GOSDT Trace (iteration = {})".format(i)

    hover_tool = HoverTool(tooltips=[
        ("name", "@name"), ("upperbound", "@upperbound"), ("lowerbound", "@lowerbound"),
        ("explored", "@explored"), ("resolved", "@resolved"), ("focused", "@focused")])
    plot.add_tools(hover_tool, TapTool(), BoxSelectTool(),BoxZoomTool(), ResetTool())

    path = "trace/{}.gml".format(i)
    with open(path) as json_file:
        data = json.load(json_file)
    G = nx.readwrite.json_graph.node_link_graph(data)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    keys = list(node for node in G)

    ### start of layout code
    y = []
    xs = {}
    x = []
    for node in G:
        support = sum( int(c) for c in node )
        y.append(support)
    
    y_min = min(y)
    y_max = max(y)
    y_step = (y_max - y_min) / 100
    for node in G:
        y_group = round((sum( int(c) for c in node ) - y_min) / max(1, y_step))
        if not y_group in xs:
            xs[y_group] = set()
        xs[y_group].add(int(node))

    groups = list(xs.keys())
    for group in groups:
        xs[group] = list(sorted(xs[group]))

    for node in G:
        x.append(int(node))

    node_sizes = [ G.nodes[node]['upperbound'] - G.nodes[node]['lowerbound'] for node in G ]
    node_lines = [ 2 if G.nodes[node]['focused'] == True else 0.5 for node in G ]
    node_colors = [ 
        '#' + 
        ('DD' if G.nodes[node]['explored'] else '77') +
        ('DD' if G.nodes[node]['resolved'] else '77') + 
        ('DD')
        for node in G ]
    edge_colors = [ d3['Category20'][20][G.edges[(src,dst)]['feature']]  for src, dst in G.edges ]

    # standardize
    min_y = min(y)
    max_y = max(y)
    width_y = (max_y - min_y) / 2
    mid_y = (max_y + min_y) / 2
    y = [ (e - mid_y) / max(1, width_y)  for e in y ]
    min_x = min(x)
    max_x = max(x)
    width_x = (max_x - min_x) / 2
    mid_x = (max_x + min_x) / 2
    x = [ (e - mid_x) / max(1, width_x) for e in x ]
    max_size = max(node_sizes)
    node_sizes = [ size / max(1, max_size) * (100-10) + 10  for size in node_sizes ]

    nx.set_node_attributes(G, dict(zip((node for node in G), node_sizes)), "node_size")
    nx.set_node_attributes(G, dict(zip((node for node in G), node_lines)), "node_line")
    nx.set_node_attributes(G, dict(zip((node for node in G), node_colors)), "node_color")
    nx.set_edge_attributes(G, dict(zip(((src, dst) for src, dst in G.edges), edge_colors)), "edge_color")

    graph_layout = dict(zip(keys, zip(x, y)))

    renderer = from_networkx(G, graph_layout, scale=1, center=(0, 0))

    renderer.node_renderer.glyph = Circle(size="node_size", fill_color="node_color", fill_alpha=0.4, line_alpha=0.7, line_width="node_line")

    if m > 0:
        renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.4, line_width=2)

    plot.renderers.append(renderer)
    return plot

i = 1
min_iter = 0
while not file_exists("trace/{}.gml".format(min_iter)):
    min_iter += 1
max_iter = min_iter
while file_exists("trace/{}.gml".format(max_iter + 1)):
    max_iter += 1

doc = curdoc()

iterator = Slider(title="Iteration", value=min_iter, start=min_iter, end=max_iter, step=1)
layout = column(iterator, replot(iterator.value))

def render(attrname, old, new):
    layout.children[1] = replot(iterator.value)

iterator.on_change('value', render)
doc.add_root(layout)