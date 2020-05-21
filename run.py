# TODO: implement a script for running the experiments, training and sampling
from analysis import experiment
from analysis import functions
from analysis import visualize

if __name__ == "__main__":
    experiment.create_class_distributions(["combined_classes.json",
                                           "orientation_classes.json",
                                           "shape_classes.json",
                                           "position_classes.json"])
    print("Created Distributions")
    experiment.create_segmentations("basic", ["simple_cuboid.json"])
    print("Created Segmentations")
    # Drawing and saving samples requires large storage space (20 GiB)
    # If samples were drawn, the training can be run without sampling on the fly
    # experiment.draw_samples()
    # experiment.run_training(False)
    # experiment.run_training(True,
    #                         ["coarse_cuboid_ENV_basic"],
    #                         ["corner_classes", "orientation_classes", "shape_classes"])
    # experiment.show_distributions("basic",
    #                               ["combined_classes"])
    # experiment.create_features(
    #     ["basic_cuboid_ENV_basic", "advanced_cuboid_ENV_basic"],
    #     ["combined_classes", "orientation_classes", "position_classes", "shape_classes"]
    # )
    # print("Finished Feature Creation")
    # experiment.run_feature_training(
    #     ["basic_cuboid_ENV_basic", "advanced_cuboid_ENV_basic"],
    #     ["combined_classes", "orientation_classes", "position_classes", "shape_classes"]
    # )
    # experiment.run_feature_training_with_variations(
    #     segmentation_sets=["basic_cuboid_ENV_basic"],
    #     single_segmentations=[["3_3_3", "4_4_4", "5_5_5", "10_10_10"]],
    #     distribution_sets=["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
    #     # cls_set="Nearest Neighbors",
    #     train_splits=[20, 30, 35, 45, 50, 70, 80, 90]
    # )
    # experiment.show_segmentation(["basic_cuboid_ENV_basic"])
    # experiment.show_feature_training_results(["basic_cuboid_ENV_basic"])
    # experiment.show_features(
    #     ["coarse_cuboid_ENV_basic"],
    #     ["shape_classes"]
    # )
    """
    functions.convert_results_to_csv(
        [
            "TS_5_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_10_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_15_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_20_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_25_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_30_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_35_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_40_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_45_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_50_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_60_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_70_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_80_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0",
            "TS_90_SEG_basic_cuboid_ENV_basic_INC_0_DEV_0"
        ],
        "comparison_training_rate"
    )
    """
    # visualize.visualize_train_data_amount_influence("results_comparison_training_rate.csv")
    experiment.create_modified_features(
        ["simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        deviation_set=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    print("Created Deviation Features")
    experiment.create_modified_features(
        ["simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        incompleteness_set=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    print("Created Incompleteness Features")
    experiment.run_feature_training_with_variations(
        segmentation_sets=["simple_cuboid_ENV_basic_INC_0_DEV_1",
                           "simple_cuboid_ENV_basic_INC_0_DEV_2",
                           "simple_cuboid_ENV_basic_INC_0_DEV_3",
                           "simple_cuboid_ENV_basic_INC_0_DEV_4",
                           "simple_cuboid_ENV_basic_INC_0_DEV_5"],
        single_segmentations=[["3_3_3", "4_4_4", "5_5_5", "10_10_10"] for _ in range(5)],
        distribution_sets=["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        cls_set=["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"],
        train_splits=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    )
    print("Trained on Deviation Features")
    experiment.run_feature_training_with_variations(
        segmentation_sets=["simple_cuboid_ENV_basic_INC_5_DEV_0",
                           "simple_cuboid_ENV_basic_INC_10_DEV_0",
                           "simple_cuboid_ENV_basic_INC_15_DEV_0",
                           "simple_cuboid_ENV_basic_INC_20_DEV_0",
                           "simple_cuboid_ENV_basic_INC_25_DEV_0"],
        single_segmentations=[["3_3_3", "4_4_4", "5_5_5", "10_10_10"] for _ in range(5)],
        distribution_sets=["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        cls_set=["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"],
        train_splits=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    )
    print("Trained on Incompleteness Features")
