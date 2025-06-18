import pandas as pd

filepath = ("p2-texts\hansard40000.csv")

def altering_party(filepath):

    data = pd.read_csv(filepath)

    # a-i)
    data['party'] = data['party'].replace({'Labour (Co-op)': 'Labour'})

def removing_values(filepath):

    data = pd.read_csv(filepath)