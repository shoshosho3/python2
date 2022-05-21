import clustering
import data
import numpy as np

if __name__ == '__main__':
    # part A
    print("Part A: ")
    df_all = data.load_data("hw1/london.csv")
    data.add_new_columns(df_all)
    data.data_analysis(df_all)

    print()

    # part B
    print("Part B: ")
    df_all = data.load_data("hw1/london.csv")
    np1 = clustering.transform_data(df_all, ["cnt", "hum"])
    print("k = 2")
    clusters = clustering.kmeans(np1, 2)
    print(np.array_str(clusters[1], precision=3, suppress_small=True))
    clustering.visualize_results(np1, clusters[0], clusters[1], "hw2/plot2")
    print("\nk = 3")
    clusters = clustering.kmeans(np1, 3)
    print(np.array_str(clusters[1], precision=3, suppress_small=True))
    clustering.visualize_results(np1, clusters[0], clusters[1], "hw2/plot3")
    print("\nk = 5")
    clusters = clustering.kmeans(np1, 5)
    print(np.array_str(clusters[1], precision=3, suppress_small=True))
    clustering.visualize_results(np1, clusters[0], clusters[1], "hw2/plot5")
