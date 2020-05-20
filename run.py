# TODO: implement a script for running the experiments, training and sampling
from analysis import experiment
from analysis import functions
from analysis import visualize

if __name__ == "__main__":
    # experiment.create_class_distributions(["combined_classes.json",
    #                                        "orientation_classes.json",
    #                                        "shape_classes.json",
    #                                        "position_classes.json"])
    experiment.create_segmentations("basic", ["simple_cuboid.json"])
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
        deviation_set=[1, 2, 3, 4, 5]
    )
    experiment.create_modified_features(
        ["simple_cuboid_ENV_basic"],
        ["combined_classes", "orientation_classes", "position_classes", "shape_classes"],
        incompleteness_set=[0.05, 0.10, 0.15, 0.20, 0.25]
    )
