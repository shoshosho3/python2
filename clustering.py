import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    transformed_data = df[features].to_numpy()
    for i in range(2):
        transformed_data[:, i] = (transformed_data[:, i] - np.min(transformed_data[:, i])) / np.sum(
            transformed_data[:, i])
    transformed_data = add_noise(transformed_data)
    return transformed_data


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    last_centroids = choose_initial_centroids(data, k)
    labels = get_labels(data, last_centroids, k)
    centroids = recompute_centroids(data, labels, k)
    while not np.array_equal(last_centroids, centroids):
        last_centroids = centroids
        labels = get_labels(data, last_centroids, k)
        centroids = recompute_centroids(data, labels, k)
    return labels, centroids


def get_labels(data, centroids, k):
    labels = np.zeros((len(data)))
    for i in range(len(data)):
        min_index = 0
        min_dist = dist(data[i], centroids[0])
        for j in range(1, k):
            this_dist = dist(data[i], centroids[j])
            if this_dist < min_dist:
                min_index = j
                min_dist = this_dist
        labels[i] = min_index
    return labels


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red', 'green', 'yellow', 'orange']
    color_fit = [colors[int(labels[i])] for i in range(len(labels))]
    plt.scatter(data[:, 0], data[:, 1], c=color_fit)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='white', edgecolors='black', marker='*')
    plt.xlabel('cnt')
    plt.ylabel('hum')
    plt.title(f'Results for kmeans with k={len(centroids)}')
    plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    return np.linalg.norm(x - y)
    # return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    pass
    # return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    new_centroids_array = np.zeros((k, 2))
    for i in range(k):
        for l in range(2):
            new_centroids_array[i][l] = np.array([data[j][l] for j in range(len(data)) if labels[j] == i]).mean()
    return new_centroids_array
    # return centroids
