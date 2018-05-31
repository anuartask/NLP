import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNN():
    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto', 
                 leaf_size=30, metric='minkowski', p=2, 
                 metric_params=None, n_jobs=1):
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, 
                                    algorithm=algorithm, 
                                    leaf_size=leaf_size, metric=metric, 
                                    p=p, metric_params=metric_params, n_jobs=n_jobs)
    
    def fit(self, X, y):
        self.knn.fit(X, y)
        self.y = y
    
    def predict_proba(self, X):
        _, neighbors = self.knn.kneighbors(X)
        probas = np.array([(self.y[neighbor, :].sum(axis=0) / self.y[neighbor, :].sum()) 
                    for neighbor in neighbors])
        return probas