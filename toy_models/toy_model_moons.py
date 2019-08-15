import numpy as np
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

class Moons():
    def __init__(self):
        self.labels_ = None
        self.data_ = None
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.name = 'moons'
        return

    def sample(self, n_points):
        self.data_, self.labels_ = datasets.make_moons(n_samples=n_points, noise=.05)
        self.classifier.fit(self.data_,self.labels_+1)
        return self.data_

    def density(self,x):
        min_dist = cdist(x,self.data_[::10]).min(axis=1)
        density = np.zeros(x.shape[0])
        density[min_dist < 5e-2] = 0.5
        density /= density.sum()
        return density

    def assign_cluster_labels(self, x):
        return self.classifier.predict(x)

