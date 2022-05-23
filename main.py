import clustering
import data
import numpy as np


def main():
    # part A
    print("Part A: ")
    df_all = data.load_data("london.csv")
    data.add_new_columns(df_all)
    data.data_analysis(df_all)

    print()

    # part B
    print("Part B: ")
    df_all = data.load_data("london.csv")
    transformed_data = clustering.transform_data(df_all, ["cnt", "hum"])
    activate_kmeans(2, transformed_data)
    print()
    activate_kmeans(3, transformed_data)
    print()
    activate_kmeans(5, transformed_data)


def activate_kmeans(k, in_data):
    """
    This function activates kmeans and saves an image representing it

    :param k: number of clusters in kmeans
    :param in_data: numpy array in shape(n, 2) of data to be clustered
    """
    print(f"k = {k}")
    clusters = clustering.kmeans(in_data, k)
    print(np.array_str(clusters[1], precision=3, suppress_small=True))
    clustering.visualize_results(in_data, clusters[0], clusters[1], f"plot{k}")


if __name__ == '__main__':
    main()
