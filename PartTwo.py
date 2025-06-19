import pandas as pd

filepath = ("p2-texts\hansard40000.csv")

data = pd.read_csv(filepath)

# a-i)
data["party"] = data["party"].replace({"Labour (Co-op)": "Labour"})

#a-ii)

data = data[data["party"] != "Speaker"]

top_parties = data["party"].value_counts().nlargest(4).index.tolist()

data = data[data["party"].isin(top_parties)]

#a-iii)

data_clean = data[:]

rows_to_drop = data_clean[data_clean["speech_class"] != "Speech"].index

data_clean = data_clean.drop(rows_to_drop)

#a-iv)

rows_to_drop = data_clean[data_clean["speech"].str.len() < 1000].index

data_clean = data_clean.drop(rows_to_drop)