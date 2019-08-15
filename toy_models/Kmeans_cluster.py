import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

class KMeansCluster():

    def __init__(self,n_min_clusters,n_max_clusters):
        self.n_min_clusters_ = n_min_clusters
        self.n_max_clusters_ = n_max_clusters
        self.labels_ = None
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.name='kmeans'
        return

    def cluster(self, x):
        print('Cluster data with K-means')
        all_cluster_labels = []
        silhouette_scores = np.zeros(self.n_max_clusters_-self.n_min_clusters_+1)

        for n_clusters in range(self.n_min_clusters_,self.n_max_clusters_+1):
            km = KMeans(n_clusters=n_clusters).fit(x)
            all_cluster_labels.append(km.labels_)
            silhouette_scores[n_clusters-self.n_min_clusters_] = silhouette_score(x, all_cluster_labels[-1])

        ind = np.argmax(silhouette_scores)
        self.labels_ = all_cluster_labels[ind]+1

        # Train kNN classifier
        self.classifier.fit(x, self.labels_)
        print('Cluster labels: '+str(np.unique(self.labels_)))
        return self.labels_

    def assign_cluster_labels(self, x):
        return self.classifier.predict(x)