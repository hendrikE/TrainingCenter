# TODO: implement a visualization for all steps
import os
import json
import pickle
import math

from analysis import functions

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx


def visualize_distributions(environment_path, distributions_path, name):

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
        subplot_titles=["Class {}".format(x) for x in range(1, no_distributions + 1)],
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
        title="Distributions '{}'".format(name),
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_segmentations(path, name):
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
        subplot_titles=["Segmentation {}".format(x) for x in range(1, no_segmentations + 1)],
        specs=specs,
        vertical_spacing=0.07
    )

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
    fig.update_layout(
        title="Segmentations '{}'".format(name),
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        showlegend=False
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
        subplot_titles=["Class {}".format(x) for x in range(1, no_distributions + 1)],
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


def confusion_matrices(path, names):
    with open(path, "rb") as matrix_file:
        matrices = pickle.load(matrix_file)

    rows = 2
    cols = 5

    specs = [[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=[x for x in matrices.keys()],
        vertical_spacing=0.07
    )
    for index, matrix in enumerate(matrices.values()):
        if index < 5:
            row = 1
            col = index + 1
        else:
            row = 2
            col = index - 4
        fig.add_trace(
            go.Heatmap(
                x=list(range(1, matrix.shape[0] + 1)),
                y=list(range(1, matrix.shape[1] + 1)),
                z=matrix,
                colorscale="Viridis",
                coloraxis="coloraxis"
            ),
            row=row,
            col=col
        )
    title = "Confusion Matrices for Different Classifiers "
    names = "Segmentation Set '{}' Segmentation {} Distributions Set '{}'".format(names[0], names[1], names[2])
    fig.update_layout(
        title=title + names,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        coloraxis=dict(colorscale='Viridis'),
        showlegend=False
    )
    fig.show()


def visualize_feature_creation(distribution, segmentation, size):
    rows = 3
    cols = 5
    specs = [
        [{"type": "scatter3d", "rowspan": 2, "colspan": 2}, None, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        [None, None, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        [None, None, {"type": "table", "rowspan": 1, "colspan": 3}, None, None]
    ]
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        # subplot_titles=["Distributions Set '{}'".format(x.split("/")[-1]) for x in paths],
        vertical_spacing=0.07,
        horizontal_spacing=0.07
    )
    distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
    sample = distribution.pdf(segmentation[:, 3:])

    fig.add_trace(
        go.Scatter3d(x=segmentation[:, 3],
                     y=segmentation[:, 4],
                     z=segmentation[:, 5],
                     mode="markers",
                     marker=dict(
                         size=np.interp(sample, (sample.min(), sample.max()), (2, 10))
                     ),
                     showlegend=False
                     ),
        row=1, col=1)

    grid = np.zeros((size[0], size[1], size[2]))
    for index, s in enumerate(segmentation[:, :3]):
        l, w, h = s
        grid[int(l), int(w), int(h)] = sample[index]
    x_dist = np.sum(grid, axis=(1, 2))
    y_dist = np.sum(grid, axis=(0, 2))
    z_dist = np.sum(grid, axis=(0, 1))

    names = ["X", "Y", "Z"]

    for index, dist in enumerate([x_dist, y_dist, z_dist]):
        fig.add_trace(
            go.Scatter(
                x=[x for x in range(1, dist.shape[0] + 1)],
                y=dist,
                mode="lines+markers",
                name="{} Distribution".format(names[index])
            ),
            row=1, col=index+3
        )

    x_cdf = np.array([np.sum(x_dist[:i + 1]) for i in range(x_dist.shape[0])])
    y_cdf = np.array([np.sum(y_dist[:i + 1]) for i in range(y_dist.shape[0])])
    z_cdf = np.array([np.sum(z_dist[:i + 1]) for i in range(z_dist.shape[0])])

    for index, dist in enumerate([x_cdf, y_cdf, z_cdf]):
        fig.add_trace(
            go.Scatter(
                x=[x for x in range(1, dist.shape[0] + 1)],
                y=dist,
                mode="lines+markers",
                name="{} Cumulated Distribution".format(names[index])
            ),
            row=2, col=index+3
        )

    features = [functions.calc_feature_values(cdf, [0.1, 0.25, 0.5, 0.75, 0.9]) for cdf in [x_cdf, y_cdf, z_cdf]]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Features", "X Dimension", "Y Dimension", "Z Dimension"],
                font=dict(size=15),
                height=30,
                align="left"
            ),
            cells=dict(
                values=np.vstack([["10 Percent", "25 Percent", "50 Percent", "75 Percent", "90 Percent"],
                                  features
                                  ]),
                font=dict(size=12),
                height=30,
                align="left")
        ),
        row=3, col=3
    )
    fig.update_layout(
        title="Features Creation",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_feature_distributions():
    pass


def visualize_feature_comparison():
    pass


def visualize_feature_accuracies(paths):
    colors = ["lightsalmon", "indianred", "darksalmon"]
    rows = 3
    cols = 1
    specs = [[{'type': 'bar'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=["Distributions Set '{}'".format(x.split("/")[-1]) for x in paths],
        vertical_spacing=0.07,
        shared_xaxes=True
    )
    for index, path in enumerate(paths):
        row = index + 1
        col = 1
        with open(os.path.join(path, "accuracies"), "rb") as accuracy_file:
            accuracies = pickle.load(accuracy_file)
        if index == 0:
            show = True
        else:
            show = False
        for seg, acc in accuracies.items():
            fig.add_trace(
                go.Bar(
                    name=seg,
                    x=[x for x in acc.keys()],
                    y=[x for x in acc.values()],
                    text=["{:.2f}".format(x) for x in acc.values()],
                    textposition="inside",
                    legendgroup="group",
                    showlegend=show,
                    marker_color=colors[int(seg) - 1]
                ),
                row=row,
                col=col
            )
    fig.update_layout(
        title="Accuracies",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_segmentation_resolution_effect():
    pass


if __name__ == "__main__":
    pass
