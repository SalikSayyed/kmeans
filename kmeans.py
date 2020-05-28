#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
digits = load_iris()
tol=0.01
inits=1
itt=400
kmeans = KMeans(n_clusters=3,max_iter=itt,tol=tol,n_init=inits,init='random')
clusters = kmeans.fit_predict(digits.data)

from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(3):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]

from sklearn.metrics import accuracy_score
score=accuracy_score(digits.target, labels)
print(score*100)
