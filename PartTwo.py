import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from time import time 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

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
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")

X = vectorizer.fit_transform(data["speech"])
y = data["party"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=26, stratify=y)

#c)

rand = RandomForestClassifier(n_estimators=300, random_state=26)
rand.fit(X_train, y_train)
rand_pred = rand.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rand_pred))

svm = SVC(kernel="linear", random_state=26)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Results:")
print(classification_report(y_test, svm_pred))

f1_score_rand = f1_score(rand_pred,y_test, average=None)
f1_score_svm = f1_score(svm_pred, y_test, average=None)
print("Random forest F1 Score:")
print(f1_score_rand)
print("SVM F1 Score:")
print(f1_score_svm)

#d)

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 3))

X = vectorizer.fit_transform(data["speech"])
y = data["party"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=26, stratify=y)

rand = RandomForestClassifier(n_estimators=300, random_state=26)
rand.fit(X_train, y_train)
rand_pred = rand.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rand_pred))

svm = SVC(kernel="linear", random_state=26)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Results:")
print(classification_report(y_test, svm_pred))

f1_score_rand = f1_score(rand_pred,y_test, average=None)
f1_score_svm = f1_score(svm_pred, y_test, average=None)
print("Random forest F1 Score:")
print(f1_score_rand)
print("SVM F1 Score:")
print(f1_score_svm)
