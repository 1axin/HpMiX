import warnings
import csv
import numpy as np
warnings.filterwarnings('ignore')

def match_features(sample_file, feature_data, feature_dim):
    def read_csv_to_list(file_name):
        data_list = []
        with open(file_name, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                data_list.append(row)
        return data_list

    samples = read_csv_to_list(sample_file)
    print(f"Loaded {len(samples)} samples from {sample_file}")
    print(f"Loaded features shape: {np.array(feature_data).shape}")

    sample_features = []
    for counter, sample in enumerate(samples):
        if counter % 100 == 0:
            print(f"Processing sample {counter}/{len(samples)}")
        node1_features = [0] * feature_dim
        node2_features = [0] * feature_dim

        for feature_row in feature_data:
            if sample[0] == str(feature_row[0]):
                node1_features = [float(x) for x in feature_row[1:]]
                break

        # Match node2 features
        for feature_row in feature_data:
            if sample[1] == str(feature_row[0]):
                node2_features = [float(x) for x in feature_row[1:]]
                break

        sample_features.append(node1_features + node2_features)

    return sample_features