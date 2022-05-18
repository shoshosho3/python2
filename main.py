import data

if __name__ == '__main__':
    df_all = data.load_data("hw2/london_sample_500.csv")
    data.add_new_columns(df_all)
    data.data_analysis(df_all)
