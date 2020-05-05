import pandas as pd
import numpy as np
from matplotlib.collections import EventCollection
import matplotlib.pyplot as plt
import pydotplus as pdot
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
%matplotlib inline

mypredictdata=pd.read_csv("~/Downloads/vix-daily_csv.csv")
mypredictdata.head(5)
a = []
x = []
y=[]
for i in range(0, len(mypredictdata)-1):
    a.append([i,mypredictdata["VIX Close"][i]])    

X = a

X = StandardScaler().fit_transform(X)

for i in range(0,len(X)):
    x.append(X[i][0])
    y.append(X[i][1])

plt.scatter(x,y)
plt.show()

db =DBSCAN(eps=0.3, min_samples=200,leaf_size=30).fit(X)
db.fit_predict(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
clusternum = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % clusternum)

unique_labels = set(labels)
plt.figure(num=None, figsize=(10, 10), facecolor='w', edgecolor='k')
for k in unique_labels:
    col=[0,0.5,1,1]
    if k == -1:
        col = [1, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', color=tuple(col),markersize=5, alpha=0.5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.title('Estimated number of clusters: %d' % clusternum)
    plt.plot(xy[:, 0], xy[:, 1], 'o', color=tuple(col), markersize=5, alpha=0.5)
