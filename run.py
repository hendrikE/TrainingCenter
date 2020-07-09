# TODO: implement a script for running the experiments, training and sampling
from analysis import experiments
from analysis import functions
from analysis import visualize

if __name__ == "__main__":

    # Sampling
    experiments.create_class_distributions(["combined_classes", "orientation_classes", "shape_classes", "position_classes"])
    experiments.show_distributions("basic", ["combined_classes", "shape_classes", "orientation_classes", "position_classes"])

    experiments.create_segmentations("basic", ["basic_cuboid", "advanced_cuboid", "simple_cuboid"])
    experiments.show_segmentation("basic", ["basic_cuboid_ENV_basic", "advanced_cuboid_ENV_basic", "simple_cuboid_ENV_basic"])

    # Create and train on features
    experiments.create_features(
        ["basic_cuboid_ENV_basic", "advanced_cuboid_ENV_basic", "simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"]
    )
    experiments.show_features(
        "basic",
        ["simple_cuboid_ENV_basic"],
        ["shape_classes", "position_classes"]
    )
    experiments.run_feature_training(
        ["basic_cuboid_ENV_basic", "advanced_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"]
    )

    # Create and train on modified features
    deviation = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    incompleteness = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    experiments.create_modified_features(
        ["simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        deviation_set=deviation
    )
    experiments.create_modified_features(
        ["simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        incompleteness_set=incompleteness
    )

    experiments.run_feature_training_with_variations(
        segmentation_sets=["simple_cuboid_ENV_basic_INC_{}_DEV_0".format(x*100) for x in incompleteness],
        single_segmentations=[["3_3_3", "4_4_4", "5_5_5", "10_10_10"] for _ in range(9)],
        distribution_sets=["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        cls_set=["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"],
        train_splits=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    )
    experiments.run_feature_training_with_variations(
        segmentation_sets=["simple_cuboid_ENV_basic_INC_0_DEV_{}".format(x) for x in deviation],
        single_segmentations=[["3_3_3", "4_4_4", "5_5_5", "10_10_10"] for _ in range(9)],
        distribution_sets=["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        cls_set=["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"],
        train_splits=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    )

    functions.convert_results_to_csv(
        ["TS_{}_SEG_simple_cuboid_ENV_basic_INC_{}_DEV_0".format(ts, inc*100)
         for ts in [10, 20, 30, 40, 50, 60, 70, 80, 90] for inc in incompleteness],
        "incompleteness"
    )
    functions.convert_results_to_csv(
        ["TS_{}_SEG_simple_cuboid_ENV_basic_INC_0_DEV_{}".format(ts, dev)
         for ts in [10, 20, 30, 40, 50, 60, 70, 80, 90] for dev in deviation],
        "deviation"
    )

    visualize.visualize_two_modifications_influence(
        "results_incompleteness.csv",
        "incompleteness",
        "training_split",
        ["deviation", 0]
    )
    visualize.visualize_two_modifications_influence(
        "results_deviation.csv",
        "deviation",
        "training_split",
        ["incompleteness", 0]
    )
