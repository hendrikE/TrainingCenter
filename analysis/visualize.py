# TODO: implement a visualization for all steps
import os
import json
import pickle
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import multivariate_normal
# import networkx as nx


def visualize_distributions(environment_path, distributions_path):

    with open(distributions_path, "r+") as dist_file:
        distributions_loaded = json.load(dist_file)

    with open(environment_path, "r+") as env_file:
        environment = json.load(env_file)

    length = environment["params"]["length"]
    width = environment["params"]["width"]
    height = environment["params"]["height"]

    distributions = [[((x[0]+x[1])/2) for x in val] for val in distributions_loaded.values()]

    x, y, z = np.mgrid[0:length:2, 0:width:2, 0:height:2]
    pos = np.empty((x.shape[0] * x.shape[1] * x.shape[2], 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()

    no_distributions = len(distributions)

    if no_distributions < 5:
        rows = 1
        cols = no_distributions
    else:
        rows = math.ceil(no_distributions / 4)
        cols = 4

    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=["Class {}".format(x) for x in range(no_distributions)],
        specs=specs,
        vertical_spacing=0.05
    )

    for index, distribution in enumerate(distributions):
        col = (index % 4) + 1
        row = (index // 4) + 1
        dist = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
        values = dist.pdf(pos)
        norm_values = ((values - min(values)) / (max(values) - min(values)))
        fig.add_trace(
            go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=norm_values,
                # isomin=-0.1,
                # isomax=0.8,
                opacity=0.1,
                surface_count=21,
            ),
            row=row,
            col=col
        )
    fig.update_layout(
        title="Distributions",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_segmentations(path):
    segmentations = os.listdir(path)
    segmentations.sort()

    no_segmentations = len(segmentations)

    if no_segmentations < 5:
        rows = 1
        cols = no_segmentations
    else:
        rows = math.ceil(no_segmentations / 4)
        cols = 4

    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=["Segmentation {}".format(x) for x in range(no_segmentations)],
        specs=specs)

    for index, segmentation in enumerate(segmentations):
        segmentation_loaded = np.load(os.path.join(path, segmentation))
        col = (index % 4) + 1
        row = (index // 4) + 1
        fig.add_trace(
            go.Scatter3d(
                x=segmentation_loaded[:, 3],
                y=segmentation_loaded[:, 4],
                z=segmentation_loaded[:, 5],
                mode='markers',
                name=str(index),
                marker=dict(
                    size=1,
                    opacity=0.8
                )
            ),
            row=row,
            col=col
        )

    fig.show()


def visualize_segmentation_over_distribution(distributions_path, segmentation_path, environment_path):
    with open(distributions_path, "r+") as dist_file:
        distributions_loaded = json.load(dist_file)

    with open(environment_path, "r+") as env_file:
        environment = json.load(env_file)

    seg = np.load(segmentation_path)

    length = environment["params"]["length"]
    width = environment["params"]["width"]
    height = environment["params"]["height"]

    distributions = [[((x[0] + x[1]) / 2) for x in val] for val in distributions_loaded.values()]

    x, y, z = np.mgrid[0:length:2, 0:width:2, 0:height:2]
    pos = np.empty((x.shape[0] * x.shape[1] * x.shape[2], 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()

    no_distributions = len(distributions)

    if no_distributions < 5:
        rows = 1
        cols = no_distributions
    else:
        rows = math.ceil(no_distributions / 4)
        cols = 4

    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=["Class {}".format(x) for x in range(no_distributions)],
        specs=specs,
        vertical_spacing=0.05
    )

    segmentation_trace = go.Scatter3d(
        x=seg[:, 3],
        y=seg[:, 4],
        z=seg[:, 5],
        mode="markers",
        marker=dict(
            size=5,
            opacity=0.2,
            colorscale='Viridis'
        )
    )

    for index, distribution in enumerate(distributions):
        col = (index % 4) + 1
        row = (index // 4) + 1
        dist = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
        values = dist.pdf(pos)
        norm_values = ((values - min(values)) / (max(values) - min(values)))
        fig.add_trace(
            go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=norm_values,
                opacity=0.1,
                surface_count=21,
            ),
            row=row,
            col=col
        )
        fig.add_trace(
            segmentation_trace,
            row=row,
            col=col
        )

    fig.update_layout(
        title="Segmentation over Distributions",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )
    fig.show()


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
        subplot_titles=[str(x) for x in range(8)],
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
    pass


def confusion_matrix(path):
    z = np.load(path)
    x = list(range(1, 9))
    y = list(range(1, 9))
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z, colorscale='Viridis')
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


if __name__ == "__main__":
    visualize_samples(
        "../analysis_files/distributions/corner_classes.json",
        "../analysis_files/segmentations/basic_cuboid_ENV_basic/1.npy"
    )
    visualize_segmentation_over_distribution(
        "../analysis_files/distributions/corner_classes.json",
        "../analysis_files/segmentations/basic_cuboid_ENV_basic/1.npy",
        "../analysis_files/environments/basic.json"
    )
