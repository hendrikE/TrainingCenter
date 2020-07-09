import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--future", help="create directories needed for future elements",
                        action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(os.path.join("analysis_files", "features")):
        os.mkdir(os.path.join("analysis_files", "features"))
        print("Created directory 'features' in directory 'analysis_files'")

    if not os.path.isdir(os.path.join("analysis_files", "feature_results")):
        os.mkdir(os.path.join("analysis_files", "feature_results"))
        print("Created directory 'feature_results' in directory 'analysis_files'")

    if args.future:
        if not os.path.isdir(os.path.join("analysis_files", "results")):
            os.mkdir(os.path.join("analysis_files", "results"))
            print("Created directory 'results' in directory 'analysis_files'")

        if not os.path.isdir(os.path.join("analysis_files", "samples")):
            os.mkdir(os.path.join("analysis_files", "samples"))
            print("Created directory 'samples' in directory 'analysis_files'")
