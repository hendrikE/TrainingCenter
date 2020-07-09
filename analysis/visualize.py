import os
import json
import pickle
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

from analysis import functions


def visualize_distributions(environment_path, distributions_path, name):
    """
    Function to visualize a set of distributions
    :param environment_path: Path to the JSON file containing information on the environment,
    in which the distributions will be placed
    :param distributions_path: Path to the JSON file containing the specifications of the distribution set
    :param name: Name of the distribution set
    :return:
    """

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
                opacity=0.1,
                surface_count=21
            ),
            row=row,
            col=col
        )
    fig.update_layout(
        height=500 * rows,
        title="<b>Distributions '{}' in Environment '{}'</b>".format(name, environment_path.split("/")[-1].split(".")[0]),
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )
    fig.show()


def sorting(element):
    """
    Function to sort segmentations in a certain way
    :param element: segmentation name (e.g. "4_4_4")
    :return:
    """
    return int(element.split("_")[0]) * int(element.split("_")[1]) * int(element.split("_")[2].split(".")[0])


def visualize_segmentations(env_path, path, name):
    """
    Function to visualize a set of segmentations
    :param env_path: Path to the JSON file containing the environment, which will be segmented
    :param path: Path to the JSON file containing the segmentation set
    :param name: Name of the segmentation set
    :return:
    """
    segmentations = os.listdir(path)
    segmentations.sort(key=sorting)

    no_segmentations = len(segmentations)

    with open(env_path, "r+") as env_file:
        environment = json.load(env_file)

    length = environment["params"]["length"]
    width = environment["params"]["width"]
    height = environment["params"]["height"]

    if no_segmentations < 5:
        rows = 1
        cols = no_segmentations
    else:
        rows = math.ceil(no_segmentations / 4)
        cols = 4

    specs = [[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=["Segmentation '{}'".format(x.split(".")[0]) for x in segmentations],
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
                    opacity=0.8,
                    color="blue"
                )
            ),
            row=row,
            col=col
        )
    scene = dict(
        xaxis=dict(
            range=[0, length]
        ),
        yaxis=dict(
            range=[0, width]
        ),
        zaxis=dict(
            range=[0, height]
        )
    )
    fig.update_layout(
        title="<b>Segmentations '{}'</b>".format(name),
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        showlegend=False
    )
    layout_dict = {}
    for i in range(len(segmentations)):
        if i > 0:
            layout_dict["scene{}".format(i + 1)] = scene
            layout_dict["scene{}_aspectmode".format(i + 1)] = "manual"
            layout_dict["scene{}_aspectratio".format(i + 1)] = dict(x=1, y=width/length, z=height/length)
        else:
            layout_dict["scene"] = scene
            layout_dict["scene_aspectmode"] = "manual"
            layout_dict["scene_aspectratio"] = dict(x=1, y=width / length, z=height / length)
    fig.update_layout(layout_dict)
    fig.show()


def visualize_segmentation_over_distribution(distributions_path, segmentation_path, environment_path):
    """
    Function to visualize a distribution set together with one segmentation
    :param distributions_path: Path to the JSON file containing the specifications of the distribution set
    :param segmentation_path: Path to a segmentation NPY file
    :param environment_path: Path to the JSON file containing the specifications of the environment
    :return:
    """
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


def confusion_matrices(path, names):
    """
    Function to visualize the confusion matrices resulting from training on one segmentation, a distribution set and
    a set of classifiers
    :param path: Path to the NPY file with the confusion matrices
    :param names: Tuple consisting of the name of the segmentation set, the name of the specific segmentation and the
    name of the distribution set
    :return:
    """
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


def visualize_feature_creation(distribution_set, distribution_name, distribution,
                               segmentation_set, segmentation_name, segmentation, size, env_path):
    """
    Function to visualize how features are created
    :param distribution_set: Name of the distribution set
    :param distribution_name: Name of the distribution
    :param distribution: Loaded distribution from JSON file
    :param segmentation_set: Name of the segmentation set
    :param segmentation_name: Name of the segmentation
    :param segmentation: Loaded segmentation from the NPY file
    :param size: Tuple with number of segments in X, Y and Z dimension
    :param env_path: Path to the JSON file containing the specifications of the environment
    :return:
    """
    with open(env_path, "r+") as env_file:
        environment = json.load(env_file)

    length = environment["params"]["length"]
    width = environment["params"]["width"]
    height = environment["params"]["height"]

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

    scene = dict(
        xaxis=dict(range=[0, length]),
        yaxis=dict(range=[0, width]),
        zaxis=dict(range=[0, height])
    )
    layout_dict = {"scene": scene,
                   "scene_aspectmode": "manual",
                   "scene_aspectratio": dict(x=1, y=width / length, z=height / length)}
    fig.update_layout(layout_dict)

    fig.update_layout(
        title="Features Creation - {} - {} - {} - {}".format(distribution_set, distribution_name,
                                                             segmentation_set, segmentation_name),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()


def visualize_feature_accuracies(paths):
    """
    Function to visualize the accuracies from a base case with one distribution set, segmentation set and classifier set
    :param paths: A list of the paths to all the pickle accuracy files
    :return:
    """
    colors = ["lightsalmon", "indianred", "darksalmon", "salmon",
              "magenta", "darkred", "palevioletred", "mediumvioletred"]
    rows = 4
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

        segmentations = list(accuracies.keys())
        segmentations.sort(key=sorting)
        for i, seg in enumerate(segmentations):
            acc = accuracies[seg]
            fig.add_trace(
                go.Bar(
                    name=seg,
                    x=[x for x in acc.keys()],
                    y=[x for x in acc.values()],
                    text=["{:.2f}".format(x) for x in acc.values()],
                    textposition="inside",
                    legendgroup="group",
                    showlegend=show,
                    marker_color=colors[int(i) - 1]
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


def visualize_one_modification_influence(results_csv, modification, unmodified):
    """
    Function to visualize the accuracies under the influence of one variable
    :param results_csv: Path to the CSV file created from the results
    :param modification: Name of the variable which is modified
    :param unmodified: Tuple for the two set variables, consisting of two tuples with name and value of the
    set variables respectively
    :return:
    """
    df = pd.read_csv(os.path.join("analysis_files", "feature_results", results_csv))
    segmentations = df["segmentation"].unique().tolist()
    modifier = df[modification].unique().tolist()
    distribution_sets = df["distribution_set"].unique().tolist()
    classifiers = df["classifier"].unique().tolist()

    for classifier in classifiers:
        colors = ["black", "indianred", "blue", "plum"]
        rows = 2
        cols = 4
        specs = [[{'type': 'scatter'} for _ in range(cols)] for _ in range(rows)]
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=specs,
            subplot_titles=["Distributions Set<br>'{}'".format(x.split("/")[-1]) for x in distribution_sets],
            vertical_spacing=0.07
        )
        for index_dist, distribution_set in enumerate(distribution_sets):
            for index_seg, segmentation in enumerate(segmentations):
                if index_dist > 0:
                    show = False
                else:
                    show = True
                fig.add_trace(
                    go.Scatter(
                        x=[5 * x for x in modifier],
                        y=df.query(
                            "classifier=='{}' and "
                            "distribution_set=='{}' and "
                            "segmentation=='{}' and "
                            "{}=='{}' and "
                            "{}=='{}'".format(classifier, distribution_set, segmentation,
                                              unmodified[0][0], unmodified[0][1],
                                              unmodified[1][0], unmodified[1][1])
                        )["accuracy"],
                        mode="lines+markers",
                        name="Segmentation '{}'".format(segmentation),
                        marker_color=colors[index_seg],
                        showlegend=show
                    ),
                    row=1,
                    col=index_dist + 1
                )

        fig.update_layout(
            title="Comparison for Different Amount of Training Data for Classifier '{}'".format(classifier),
            xaxis_title="# train samples",
            xaxis2_title="# train samples",
            xaxis3_title="# train samples",
            xaxis4_title="# train samples",
            yaxis_title="accuracy",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        fig.update_yaxes(
            range=[0.0, 1.2]
        )
        fig.show()


def visualize_two_modifications_influence(results_csv, modification_one, modification_two, unmodified):
    """
    Function to visualize the accuracies under the influence of two variables
    :param results_csv: Path to the CSV file created from the results
    :param modification_one: Name of the first variable
    :param modification_two: Name of the second variable
    :param unmodified: Tuple containing name and value of the set variable
    :return:
    """
    df = pd.read_csv(os.path.join("analysis_files", "feature_results", results_csv))
    segmentations = df["segmentation"].unique().tolist()
    modifier_one = df[modification_one].unique().tolist()
    modifier_two = df[modification_two].unique().tolist()
    distribution_sets = df["distribution_set"].unique().tolist()
    classifiers = df["classifier"].unique().tolist()

    for classifier in classifiers:
        rows = 4
        cols = 4
        specs = [[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)]
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=specs,
            x_title="<b>Distribution Sets</b>",
            y_title="<b>Segmentations</b>",
            column_titles=distribution_sets,
            row_titles=segmentations,
            # subplot_titles=["Distributions Set <b>'{}'</b><br>Segmentation <b>'{}'</b>".format(dist, seg)
            #                 for dist in distribution_sets for seg in segmentations],
            vertical_spacing=0.005
        )
        for index_dist, distribution_set in enumerate(distribution_sets):
            for index_seg, segmentation in enumerate(segmentations):
                z = []
                for two in modifier_two:
                    z.append(
                        df.query(
                            "classifier=='{}' and "
                            "distribution_set=='{}' and "
                            "segmentation=='{}' and "
                            "{}=='{}' and "
                            "{}=='{}'".format(classifier, distribution_set, segmentation,
                                              unmodified[0], unmodified[1],
                                              modification_two, two)
                        )["accuracy"]
                    )
                z = np.array(z)
                fig.add_trace(
                    go.Surface(
                        x=modifier_one,
                        y=[x * 5 for x in modifier_two],
                        z=z,
                        colorscale="Viridis",
                        coloraxis="coloraxis"
                    ),
                    row=index_seg + 1,
                    col=index_dist + 1
                )
        if modification_one == "deviation":
            x_title_text = "<b>{} [cm]</b>".format(modification_one)
        else:
            x_title_text = "<b>{} [%]</b>".format(modification_one)
        scene = dict(
                xaxis=dict(
                    title_text=x_title_text,
                    title_font={"size": 12},
                    # range=[modifier_one[0] - ((modifier_one[-1] - modifier_one[0]) / 10),
                    #        modifier_one[-1] + ((modifier_one[-1] - modifier_one[0]) / 10)],
                    tickmode="array",
                    tickvals=modifier_one,
                    tickfont=dict(
                        size=12,
                        family="Courier New, monospace",
                    )
                ),
                yaxis=dict(
                    # title_text="<b>{}</b>".format(modification_two),
                    title_text="<b># train samples</b>",
                    title_font={"size": 12},
                    # range=[modifier_two[0] - ((modifier_two[-1] - modifier_two[0]) / 10),
                    #        modifier_two[-1] + ((modifier_two[-1] - modifier_two[0]) / 10)],
                    tickmode="array",
                    tickvals=[x * 5 for x in modifier_two],
                    tickfont=dict(
                        size=12,
                        family="Courier New, monospace",
                    ),
                ),
                zaxis=dict(
                    title_text="<b>accuracy</b>",
                    title_font={"size": 12},
                    range=[0, 1.2],
                    tickmode="array",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                    tickfont=dict(
                        size=12,
                        family="Courier New, monospace",
                    ),
                )
            )
        camera = dict(
            eye=dict(x=1.85, y=1.85, z=1.85)
        )
        layout_dict = {}
        for i in range(len(segmentations)):
            if i > 0:
                layout_dict["scene{}".format(i + 1)] = scene
                layout_dict["scene{}_camera".format(i + 1)] = camera
            else:
                layout_dict["scene"] = scene
                layout_dict["scene_camera"] = camera
        fig.update_layout(layout_dict)
        fig.update_layout(
            height=1800,
            width=1800,
            xaxis_title=modification_one,
            yaxis_title=modification_two,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ),
            coloraxis=dict(colorscale='Viridis', cmin=0.0, cmax=1.0),
            showlegend=False
        )
        fig.update_layout(
            title=dict(
                text="<b>Accuracy Comparison for Varying '{}' and '{}' "
                     "with '{}' - '{}' and Classifier '{}'</b>".format(modification_one, modification_two,
                                                                       unmodified[0], unmodified[1], classifier),
                yref="paper",
                yanchor="top",
                font={"size": 20}
            )
        )
        fig.show()
