# TODO: implement a script for running the experiments, training and sampling
from analysis import experiment

if __name__ == "__main__":
    # experiment.create_class_distributions()
    # experiment.create_segmentations("basic")
    # Drawing and saving samples requires large storage space (20 GiB)
    # If samples were drawn, the training can be run without sampling on the fly
    # experiment.draw_samples()
    # experiment.run_training(False)
    experiment.run_training(True, ["basic_cuboid_ENV_basic"], ["corner_classes"])
