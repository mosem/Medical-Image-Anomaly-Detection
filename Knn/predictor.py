from sklearn.neighbors import NearestNeighbors

def predict(train_set, test_set):
    train_feature_space, train_labels_space = train_set
    test_feature_space, test_labels_space = test_set
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(train_feature_space)
    correct_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for sample, label in zip(test_feature_space, test_labels_space):
        nearest_neighbor_index = neigh.kneighbors([sample], return_distance=False)
        if train_labels_space[nearest_neighbor_index] == label:
            correct_count += 1
            if label == 0:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if label == 0:
                false_negative += 1
            else:
                false_positive += 1

    n_samples = len(test_labels_space)
    accuracy = correct_count / n_samples
    tpr = true_positive / n_samples
    fpr = false_positive / n_samples
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    return accuracy, tpr, fpr, recall, precision
