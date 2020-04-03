# TODO: implement a visualization for all steps
import plotly.graph_objects as go
import numpy as np
from scipy.stats import multivariate_normal


def visualize_distributions(dist, dim):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=dist[:, 0],
            y=dist[:, 1],
            z=dist[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                opacity=0.8
            )
        )
    )
    fig.update_xaxes(range=[0, dim[0]])
    fig.update_yaxes(range=[0, dim[1]])
    fig.update_zaxes(range=[0, dim[2]])
    fig.show()


def visualize_segmentations(dim):
    x, y, z = np.mgrid[0:100:1, 0:100:1, 0:100:1]
    pos = np.empty((100*100*100, 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()
    dist = multivariate_normal([50, 50, 50], [[30, 0, 0], [0, 30, 0], [0, 0, 30]])
    values = dist.pdf(pos)
    norm_values = ((values-min(values))/(max(values)-min(values)))
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=norm_values,
            # isomin=-0.1,
            # isomax=0.8,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=21,  # needs to be a large number for good volume rendering
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=dim[:, 0],
            y=dim[:, 1],
            z=dim[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.8
            )
        )
    )
    fig.show()


def visualize_seg_over_dist(dist, seg, dim):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=dist[:, 0],
            y=dist[:, 1],
            z=dist[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                opacity=0.8
            )
        )
    )
    fig.update_xaxes(range=[0, dim[0]])
    fig.update_yaxes(range=[0, dim[1]])
    fig.update_zaxes(range=[0, dim[2]])
    fig.show()


def visualize_samples(sample, dim):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sample[:, 0],
            y=sample[:, 1],
            z=sample[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=sample[:, 3],
                opacity=0.8
            )
        )
    )
    fig.update_xaxes(range=[0, dim[0]])
    fig.update_yaxes(range=[0, dim[1]])
    fig.update_zaxes(range=[0, dim[2]])
    fig.show()


def visualize_training(history):
    pass


def confusion_matrix(labels, predictions):
    pass


if __name__ == "__main__":
    visualize_segmentations((np.array(([30, 30, 30], [70, 70, 70]))))
