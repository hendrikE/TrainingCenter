import json
import math
import pickle

import networkx as nx
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal


def visualize_samples(distribution_path, segmentation_path):

    with open(distribution_path, "r+") as dist_file:
        distribution_set_loaded = json.load(dist_file)

    distributions = [[((x[0] + x[1]) / 2) for x in val] for val in distribution_set_loaded.values()]

    no_distributions = len(distributions)

    if no_distributions < 5:
        rows = 1
        cols = no_distributions
    else:
        rows = math.ceil(no_distributions / 4)
        cols = 4

    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    seg = np.load(segmentation_path)

    fig = make_subplots(
        rows=rows, cols=cols,
        vertical_spacing=0.05,
        subplot_titles=["Class {}".format(x) for x in range(1, no_distributions + 1)],
        specs=specs)

    for index, val in enumerate(distributions):
        col = (index % 4) + 1
        row = (index // 4) + 1
        dist = multivariate_normal(val[:3], np.diag(val[3:]))
        sample = dist.pdf(seg[:, 3:])

        fig.add_trace(
            go.Scatter3d(x=seg[:, 3],
                         y=seg[:, 4],
                         z=seg[:, 5],
                         mode="markers",
                         marker=dict(
                             size=5,
                             color=sample,
                             opacity=0.2,
                             colorscale='Viridis'
                         )),
            row=row, col=col)
    fig.update_layout(
        title="Samples",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_training(path):
    with open(path, "rb") as file:
        history = pickle.load(file)

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter'}],
               [{'type': 'scatter'}]],
        vertical_spacing=0.01,
        shared_xaxes=True
    )
    fig.add_trace(go.Scatter(x=list(range(1, len(history["accuracy"]) + 1)), y=history["accuracy"],
                             mode='lines+markers',
                             name='Accuracy'), 1, 1)
    fig.add_trace(go.Scatter(x=list(range(1, len(history["val_accuracy"]) + 1)), y=history["val_accuracy"],
                             mode='lines+markers',
                             name='Validation Accuracy'), 1, 1)
    fig.add_trace(go.Scatter(x=list(range(1, len(history["loss"]) + 1)), y=history["loss"],
                             mode='lines+markers',
                             name='Loss'), 2, 1)
    fig.add_trace(go.Scatter(x=list(range(1, len(history["val_loss"]) + 1)), y=history["val_loss"],
                             mode='lines+markers',
                             name='Validation Loss'), 2, 1)

    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(
        title="Training History",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


def plot_keras_graph():
    path = "../analysis_files/results/coarse_cuboid_ENV_basic/center_classes/segmentation_1/model_graph"
    with open(path, "rb") as file:
        graph = pickle.load(file)
    graph = nx.Graph(nx.nx_pydot.from_pydot(graph))