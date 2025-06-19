import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time 

filepath = ("p2-texts\hansard40000.csv")

data = pd.read_csv(filepath)

# a-i)
data["party"] = data["party"].replace({"Labour (Co-op)": "Labour"})

#a-ii)

data = data[data["party"] != "Speaker"]

top_parties = data["party"].value_counts().nlargest(4).index.tolist()

data = data[data["party"].isin(top_parties)]

#a-iii)

data = data[data['speech_class'] == 'Speech']    


#a-iv)

data = data[data['speech'].str.len() >= 1000]  

#b)
"t0 = time()"
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")