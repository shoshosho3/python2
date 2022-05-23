import pandas as pd
from datetime import datetime


def load_data(path):
    """
    This function loads data from path
    :param path: path of data
    :return: data frame with all data in path
    """
    data_frame = pd.read_csv(path)
    return data_frame


def add_new_columns(df):
    """
    This function adds new columns to data frame

    :param df: data frame with data
    """
    df["season_name"] = df["season"].apply(get_name)
    df["Hour"] = df["timestamp"].apply(time, args=["%H"])
    df["Day"] = df["timestamp"].apply(time, args=["%d"])
    df["Month"] = df["timestamp"].apply(time, args=["%m"])
    df["Year"] = df["timestamp"].apply(time, args=["%Y"])
    df["is_weekend_holiday"] = df.apply(lambda x: x.is_weekend + 2 * x.is_holiday, axis=1)
    df["t_diff"] = df.apply(lambda x: x.t2 - x.t1, axis=1)


def time(timestamp, time_format):
    return int(datetime.strptime(timestamp, "%d/%m/%Y %H:%M").strftime(time_format))


def get_name(num):
    """
    This function returns season according to given number
    :param num: number of season
    :return: season according to given number(0-3 starting with spring)
    """
    if num == 0:
        return "spring"
    if num == 1:
        return "summer"
    if num == 2:
        return "fall"
    return "winter"


def data_analysis(df):
    """
    This function prints analysis of given data

    :param df: data frame with data to be analyzed
    """
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    # dictionary with correlations by 2 features in data
    corr_dictionary = {(col1, col2): "%.6f" % corr[col1][col2] if corr[col1][col2] > 0 else "%.6f" % -corr[col1][col2]
                       for col1 in corr.columns for col2 in corr.columns[corr.columns.get_loc(col1) + 1:] if
                       corr[col1][col2]}
    # sorted dictionary
    sorted_dictionary = sorted(corr_dictionary.items(), key=lambda x: x[1])
    # prints of 5 highest and 5 lowest
    print("Highest correlated are: ")
    dic_len = len(sorted_dictionary)
    for i in range(5):
        print(f"{i + 1}. {sorted_dictionary[dic_len - i - 1][0]} with {sorted_dictionary[dic_len - i - 1][1]}")
    print("\nLowest correlated are: ")
    for i in range(5):
        print(f"{i + 1}. {sorted_dictionary[i][0]} with {sorted_dictionary[i][1]}")

    print()

    # printing average t_diff by season
    group = df[["season_name", "t_diff"]].groupby(['season_name']).mean().to_dict()
    for item1 in group.items():
        for key, value in item1[1].items():
            val = "%.2f" % value
            print(f"{key} average t_diff is {val}")
    all_mean = "%.2f" % df[["t_diff"]].mean().to_dict()["t_diff"]
    print(f"All average t_diff is {all_mean}")
