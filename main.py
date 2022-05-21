import clustering
import data
import numpy as np

if __name__ == '__main__':

    # part A
    print("Part A: ")
    df_all = data.load_data("hw2/london_sample_500.csv")
    data.add_new_columns(df_all)
    data.data_analysis(df_all)

    print()

    # part B
    print("Part B: ")
    df_all = data.load_data("hw2/london_sample_500.csv")
    np1 = clustering.transform_data(df_all, ["cnt", "hum"])
    print("k = 2")
    print(np.array_str(clustering.kmeans(np1, 2)[1], precision=3, suppress_small=True))
    #print(clustering.kmeans(np, 2)[1])
    print("\nk = 3")
    #print(clustering.kmeans(np1, 3)[1])
    print(np.array_str(clustering.kmeans(np1, 3)[1], precision=3, suppress_small=True))
    print("\nk = 5")
    #print(clustering.kmeans(np1, 5)[1])
    print(np.array_str(clustering.kmeans(np1, 5)[1], precision=3, suppress_small=True))
