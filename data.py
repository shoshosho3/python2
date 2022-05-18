import pandas as pd
from datetime import datetime


def load_data(path):
    data_frame = pd.read_csv(path)
    return data_frame


def add_new_columns(df):
    df["season_name"] = df["season"].apply(get_name)
    df["Hour"] = df["timestamp"].apply(lambda x: int(datetime.strptime(x, "%d/%m/%Y %H:%M").strftime("%H")))
    df["Day"] = df["timestamp"].apply(lambda x: int(datetime.strptime(x, "%d/%m/%Y %H:%M").strftime("%d")))
    df["Month"] = df["timestamp"].apply(lambda x: int(datetime.strptime(x, "%d/%m/%Y %H:%M").strftime("%m")))
    df["Year"] = df["timestamp"].apply(lambda x: int(datetime.strptime(x, "%d/%m/%Y %H:%M").strftime("%Y")))
    df["is_weekend_holiday"] = df["is_weekend"] + df["is_holiday"].apply(lambda x: 2 * x)
    df["t_diff"] = df["t2"] - df["t1"]


def get_name(num):
    if num == 0:
        return "spring"
    if num == 1:
        return "summer"
    if num == 2:
        return "fall"
    return "winter"


def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()
    corr_dictionary = {(col1, col2): "%.6f" % corr[col1][col2] if corr[col1][col2] > 0 else "%.6f" % -corr[col1][col2]
                       for col1 in corr.columns for col2 in corr.columns[corr.columns.get_loc(col1) + 1:] if
                       corr[col1][col2]}
    sorted_dictionary = sorted(corr_dictionary.items(), key=lambda x: x[1])
    print("Highest correlated are: ")
    dic_len = len(sorted_dictionary)
    for i in range(5):
        print(f"{i+1}. {sorted_dictionary[dic_len - i - 1][0]} with {sorted_dictionary[dic_len - i - 1][1]}")
    print("Lowest correlated are: ")
    for i in range(5):
        print(f"{i+1}. {sorted_dictionary[i][0]} with {sorted_dictionary[i][1]}")

    group = df[["season_name", "t_diff"]].groupby(['season_name']).mean().to_dict()
    for item1 in group.items():
        for key, value in item1[1].items():
            val = "%.2f" % value
            print(f"{key} average t_diff is {val}")
    all_mean = "%.2f" % df[["t_diff"]].mean().to_dict()["t_diff"]
    print(f"All average t_diff is {all_mean}")
