import random

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal


def paths_through_segmentation():
    """
    Function to visualize a route through a 3x3x3 and 20x3x3 segmentation
    :return:
    """
    centers_one = []
    length_part, width_part, height_part = [3, 3, 3]
    length, width, height = [100, 100, 100]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for h in range(height_part):
        if h % 2 == 1:
            for w in range(width_part):
                if w % 2 == 1:
                    for l in range(length_part):
                        centers_one.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
                else:
                    for l in range(length_part - 1, -1, -1):
                        centers_one.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
        else:
            for w in range(width_part - 1, -1, -1):
                if w % 2 == 0:
                    for l in range(length_part):
                        centers_one.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
                else:
                    for l in range(length_part - 1, -1, -1):
                        centers_one.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
    centers_one = np.array(centers_one)

    centers_two = []
    length_part, width_part, height_part = [20, 3, 3]
    length, width, height = [100, 100, 100]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for h in range(height_part):
        if h % 2 == 1:
            for w in range(width_part):
                if w % 2 == 1:
                    for l in range(length_part):
                        centers_two.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
                else:
                    for l in range(length_part - 1, -1, -1):
                        centers_two.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
        else:
            for w in range(width_part - 1, -1, -1):
                if w % 2 == 0:
                    for l in range(length_part):
                        centers_two.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
                else:
                    for l in range(length_part - 1, -1, -1):
                        centers_two.append([
                            l,
                            w,
                            h,
                            seg_length * (l + 0.5),
                            seg_width * (w + 0.5),
                            seg_height * (h + 0.5)
                        ])
    centers_two = np.array(centers_two)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Segmentation 3x3x3", "Segmentation 20x3x3"],
        specs=[[{'type': 'scatter3d'} for _ in range(2)]],
        vertical_spacing=0.07
    )

    fig.add_trace(
        go.Scatter3d(
            x=centers_one[:, 3],
            y=centers_one[:, 4],
            z=centers_one[:, 5],
            mode='lines+markers',
            name="Missing",
            marker=dict(
                size=1,
                opacity=0.8,
                color="red"
            ),
            showlegend=False
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=centers_two[:, 3],
            y=centers_two[:, 4],
            z=centers_two[:, 5],
            mode='lines+markers',
            name="Missing",
            marker=dict(
                size=1,
                opacity=0.8,
                color="red"
            ),
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title="<b>Robot Paths Through Segmentations</b>",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        height=500,
        width=800
    )
    scene = dict(
        xaxis=dict(range=[0, length]),
        yaxis=dict(range=[0, width]),
        zaxis=dict(range=[0, height])
    )

    layout_dict = {"scene2": scene,
                   "scene2_aspectmode": "manual",
                   "scene2_aspectratio": dict(x=1, y=width / length, z=height / length),
                   "scene": scene,
                   "scene_aspectmode": "manual",
                   "scene_aspectratio": dict(x=1, y=width / length, z=height / length)}

    fig.update_layout(layout_dict)
    fig.show()


def visualize_segmentation_deviation():
    """
    Function to visualize how the segmentation looks like with different grades of deviation
    :return:
    """
    centers = []
    length_part, width_part, height_part = [5, 5, 5]
    length, width, height = [100, 100, 100]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for l in range(length_part):
        for w in range(width_part):
            for h in range(height_part):
                centers.append([
                    l,
                    w,
                    h,
                    seg_length * (l + 0.5),
                    seg_width * (w + 0.5),
                    seg_height * (h + 0.5)
                ])
    centers = np.array(centers)

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Deviation '{} cm'".format(x) for x in [1, 2, 3, 4]],
        specs=[[{'type': 'scatter3d'} for _ in range(4)]],
        vertical_spacing=0.07
    )

    for index, deviation in enumerate([1, 2, 3, 4]):
        centers_new = []
        for index_two in range(centers.shape[0]):
            centers_new.append(multivariate_normal(centers[index_two, 3:], np.diag([deviation for _ in range(3)])).rvs())
        centers_new = np.array(centers_new)
        if index == 0:
            show = True
        else:
            show = False
        fig.add_trace(
            go.Scatter3d(
                x=centers_new[:, 0],
                y=centers_new[:, 1],
                z=centers_new[:, 2],
                mode='markers',
                name="Changed",
                marker=dict(
                    size=1,
                    opacity=0.8,
                    color="red"
                ),
                showlegend=show
            ),
            row=1,
            col=index + 1
        )
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 3],
                y=centers[:, 4],
                z=centers[:, 5],
                mode='markers',
                name="Unchanged",
                marker=dict(
                    size=1,
                    opacity=0.8,
                    color="blue"
                ),
                showlegend=show
            ),
            row=1,
            col=index + 1
        )

    fig.update_layout(
        title="<b>Segmentation 5x5x5 for Different Deviations</b>",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        height=500
    )
    fig.show()


def visualize_segmentation_incompleteness():
    """
    Function to visualize how the segmentation looks like with different grades of incompleteness
    :return:
    """
    centers = []
    length_part, width_part, height_part = [5, 5, 5]
    length, width, height = [100, 100, 100]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for l in range(length_part):
        for w in range(width_part):
            for h in range(height_part):
                centers.append([
                    l,
                    w,
                    h,
                    seg_length * (l + 0.5),
                    seg_width * (w + 0.5),
                    seg_height * (h + 0.5)
                ])
    centers = np.array(centers)

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Incompleteness '{} %'".format(x) for x in [10, 20, 30, 40]],
        specs=[[{'type': 'scatter3d'} for _ in range(4)]],
        vertical_spacing=0.07
    )

    for index, incompleteness in enumerate([10, 20, 30, 40]):
        centers_left = []
        centers_missing = []
        for index_two in range(centers.shape[0]):
            if random.random() < incompleteness/100:
                centers_missing.append(centers[index_two, 3:])
            else:
                centers_left.append(centers[index_two, 3:])
        centers_left = np.array(centers_left)
        centers_missing = np.array(centers_missing)
        if index == 0:
            show = True
        else:
            show = False
        fig.add_trace(
            go.Scatter3d(
                x=centers_missing[:, 0],
                y=centers_missing[:, 1],
                z=centers_missing[:, 2],
                mode='markers',
                name="Missing",
                marker=dict(
                    size=1,
                    opacity=0.8,
                    color="red"
                ),
                showlegend=show
            ),
            row=1,
            col=index + 1
        )
        fig.add_trace(
            go.Scatter3d(
                x=centers_left[:, 0],
                y=centers_left[:, 1],
                z=centers_left[:, 2],
                mode='markers',
                name="Left-Over",
                marker=dict(
                    size=1,
                    opacity=0.8,
                    color="blue"
                ),
                showlegend=show
            ),
            row=1,
            col=index + 1
        )

    fig.update_layout(
        title="<b>Segmentation 5x5x5 for Different Incompleteness Probabilities</b>",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        height=500
    )
    fig.show()


def show_incompleteness_removal():
    """
    Function to visualize how incompleteness is dealt with
    :return:
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            y=[3.5, 4.5, 5.5, 3.5, 4.5, 5.5, 3.5, 4.5, 5.5],
            text=["Out of Grid", "Out of Grid", "Out of Grid", "1.5", "<b>Missing</b>", "1.5", "Missing", "3.0", "2.0"],
            mode="text"
        )
    )

    shapes = []
    for x in [-1, 0, 1]:
        for y in [3, 4, 5]:
            if x == 0 and y == 4:
                color = "grey"
            elif x == -1 or (x == 1 and y == 3):
                color = "red"
            else:
                color = "green"
            shapes.append(
                dict(type="rect", xref="x1", yref="y1",
                     x0=x, y0=y, x1=x + 1, y1=y + 1,
                     fillcolor=color, opacity=0.2)
            )
    shapes.append(
        dict(type="line", xref="x1", yref="y1",
             x0=0, y0=2, x1=0, y1=7)
    )

    fig.update_layout(
        shapes=shapes,
        title="<b>Calculating Replacement for Missing Value (0.5, 4.5)</b>",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        width=900,
        height=900,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    fig.update_xaxes(range=[-2, 3])
    fig.update_yaxes(range=[2, 7])

    fig.show()
