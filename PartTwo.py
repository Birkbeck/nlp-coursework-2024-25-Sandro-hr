import pandas as pd

filepath = ("p2-texts\hansard40000.csv")

data = pd.read_csv(filepath)

# a-i)
data["party"] = data["party"].replace({"Labour (Co-op)": "Labour"})

#a-ii)

data = data[data["party"] != "Speaker"]

top_parties = data["party"].value_counts().nlargest(4).index.tolist()

data = data[data["party"].isin(top_parties)]